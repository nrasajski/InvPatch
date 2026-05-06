import torch
import nltk
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import hamming

from Levenshtein import distance
from dataset import InvPatchMUGENTrainDataset, InvPatchMUGENTestDataset
from dataset import InvPatchMUGENTrainDataset_32_Frames, InvPatchMUGENTestDataset_32_Frames
from custom_tokenizer import ActionTokenizerMUGEN
from models.model import InvPatch
from typing import Any
from time import time
import json
from pathlib import Path
import importlib
import math
import numpy as np
from exp_configurations import *


def save_dictionary(path, params_dict):
    with open(path, 'w') as file:
        json.dump(params_dict, file, indent=4)


def train_decoder(train_model, data_loader, num_epochs, learning_rate, ignore_index, smoothing_factor, warmup_epochs,
                  is_multiframe=False, weight_decay: float = 0.0):
    train_model.train()
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=smoothing_factor)

    optimizer = AdamW(train_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / num_epochs  # Linear warmup
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)  # Normalize cosine phase
            return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

    # Create the scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(num_epochs):
        pbar = tqdm(data_loader, desc=f"Epoch: {epoch}")
        for batch in pbar:
            with torch.no_grad():
                video_latent, custom_tokens, action_string = batch
                action_embedding = video_latent.to(device)
                action_tokens = custom_tokens.to(device)
            
            if is_multiframe:
                outputs, prefix_projections = train_model.forward_multiframe(action_embedding, action_tokens)
                logits = outputs.logits
                logits = outputs.logits[:, prefix_projections.shape[1] - 1: -1]
            else:
                outputs, _ = train_model(action_embedding, action_tokens)
                logits = outputs.logits
                logits = logits[:, : -1]
                
            logits = logits.reshape(-1, logits.shape[-1])
            action_tokens = action_tokens.flatten()

            loss = loss_ce(logits, action_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': f"{loss.item()}"})
        scheduler.step()
    return train_model


def evaluate(evaluation_model,
             data_loader,
             temperature: float = 1.0,
             is_multiframe=False,
             verbose: bool = False):
    total_levenstein_distance = 0
    total_bleu_score_1 = 0
    total_bleu_score_2 = 0
    total_bleu_score_3 = 0
    total_bleu_score_4 = 0
    total_meteor_score = 0
    total_f1_macro_score = 0
    y_pred = []
    y_true = []

    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    l1 = torch.nn.L1Loss()
    l2 = torch.nn.MSELoss()
    if verbose:
        iterator_loader = data_loader
    else:
        desc = f"Running evaluation"
        iterator_loader = tqdm(data_loader, desc=desc)

    with torch.no_grad():
        for batch in iterator_loader:
            video_latent, action_string, file_name = batch
            video_latent = video_latent.to(device)
            content_embedding = video_latent

            generated_text = evaluation_model.decode(content_embedding, temperature=temperature, max_tokens=MAX_CONTEXT_LEN)
                
            action_string = action_string[0].strip().replace(" , ", ",").replace(", ", ",").lower()
            
            if verbose:
                print(f"Generated text: {generated_text} "
                        f"Actual text: {action_string} ")

            generated_array = generated_text.split(",")
            action_array = action_string.split(",")

            bleu_score_1 = sentence_bleu(references=[action_array], hypothesis=generated_array, weights=[1.],
                                         smoothing_function=SmoothingFunction().method4)
            bleu_score_2 = sentence_bleu(references=[action_array], hypothesis=generated_array,
                                         weights=[1. / 2., 1. / 2.],
                                         smoothing_function=SmoothingFunction().method4)
            bleu_score_3 = sentence_bleu(references=[action_array], hypothesis=generated_array,
                                         weights=[1. / 3., 1. / 3., 1. / 3.],
                                         smoothing_function=SmoothingFunction().method4)
            bleu_score_4 = sentence_bleu(references=[action_array], hypothesis=generated_array,
                                         smoothing_function=SmoothingFunction().method4)
            meteor_score = single_meteor_score(hypothesis=generated_array, reference=action_array)

            
            total_bleu_score_1 += bleu_score_1
            total_bleu_score_2 += bleu_score_2
            total_bleu_score_3 += bleu_score_3
            total_bleu_score_4 += bleu_score_4
            total_meteor_score += meteor_score

            label_map = _tokenizer.get_token_to_id()
            padding_token = _tokenizer.pad_token
            try:
                if len(generated_array) < len(action_array):
                    generated_array = generated_array + [padding_token] * (len(action_array) - len(generated_array))
                elif len(generated_array) > len(action_array):
                    action_array = action_array + [padding_token] *  (len(generated_array) - len(action_array))
            
            
                y_pred_int = [label_map[word] for word in generated_array]
                y_true_int = [label_map[word] for word in action_array]
                y_pred.extend(y_pred_int)
                y_true.extend(y_true_int)
            except Exception as e:
                print(e)
                print("Error in validation")
                continue

    total_samples = len(data_loader)
    return {
        "BLEU@1": f"{total_bleu_score_1 / total_samples}",
        "BLEU@2": f"{total_bleu_score_2 / total_samples}",
        "BLEU@3": f"{total_bleu_score_3 / total_samples}",
        "BLEU@4": f"{total_bleu_score_4 / total_samples}",
        "Meteor": f"{total_meteor_score / total_samples}",
        "F1_macro": f"{f1_score(y_true, y_pred, average='macro')}",
        "F1_micro": f"{f1_score(y_true, y_pred, average='micro')}",
        "F1_weighted": f"{f1_score(y_true, y_pred, average='weighted')}",
        "Precision_micro": f"{precision_score(y_true, y_pred, average='micro')}",
        "Precision_macro": f"{precision_score(y_true, y_pred, average='macro')}",
        "Precision_weighted": f"{precision_score(y_true, y_pred, average='weighted')}",
        "Recall_micro": f"{recall_score(y_true, y_pred, average='micro')}",
        "Recall_macro": f"{recall_score(y_true, y_pred, average='macro')}",
        "Recall_weighted": f"{recall_score(y_true, y_pred, average='weighted')}",
        "Accuracy": f"{accuracy_score(y_true, y_pred)}",
        "Wasserstein": f"{wasserstein_distance(y_true, y_pred)}",
        "Hamming": f"{hamming(y_true, y_pred)}"
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nltk.download('wordnet')

    PROJECT_ROOT = "PATH_TO_PROJECT_ROOT"
    RUNS_SAVE_PATH = f'{PROJECT_ROOT}/idm/idm_mugen/artifacts/runs'
    exp_configurations = [
        *some_configuration
    ]
    for configuration in exp_configurations:

        NUM_FRAMES = configuration["frames"]
        VISION_MODEL = configuration["backbone"]
        METHOD = configuration["model"]
        ACTION_FILE_PATH = configuration["actions_file"]
        SPLIT_FOLDER = configuration["split_folder"]
        SPLIT_FOLD = configuration["split_num"]
        MAX_CONTEXT_LEN = configuration["seq_len"]
        LEARNING_RATE = configuration["lr"]
        WEIGHT_DECAY = configuration["weight_decay"]
        SMOOTHING_FACTOR = configuration["smoothing"]
        PREFIX_SIZE = configuration["backbone_emb_dim"]
        POS_2D_W = configuration["2d_w"]
        POS_2D_H = configuration["2d_h"]
        DECODER_EMB_SIZE = configuration["decoder_emb_size"]
        DECODER_HEADS = configuration["decoder_heads"]
        DECODER_LAYERS = configuration["decoder_layers"]
        NUM_EPOCHS = configuration["num_epochs"]
        WARMUP_EPOCHS = configuration["warmup_epochs"]
        TRAIN_BATCH_SIZE = configuration["train_batch_size"]
        TEST_BATCH_SIZE = configuration["test_batch_size"]
        TEMPERATURE = configuration["temperature"]

        TRAIN_SET = f"{PROJECT_ROOT}/idm/idm_mugen/artifacts/dataset_splits/{SPLIT_FOLDER}/train_{SPLIT_FOLD}.json"
        TEST_SET = f"{PROJECT_ROOT}/idm/idm_mugen/artifacts/dataset_splits/{SPLIT_FOLDER}/test_{SPLIT_FOLD}.json"
        
        _tokenizer = ActionTokenizerMUGEN(context_len=MAX_CONTEXT_LEN)
        
        IGNORE_INDEX = 0

        # Evaluation parameters
        WEIGHTS_DIR = f"{RUNS_SAVE_PATH}/{VISION_MODEL}/{METHOD}_{NUM_FRAMES}_FRAMES/{round(time())}"
        Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
		if NUM_FRAMES == 32:
            dataset_train = TextDecoderMUGEN_32_Frames(splits_path=TRAIN_SET, tokenizer=_tokenizer, actions_file=actions_file_path)
        else:
            dataset_train = TextDecoderMUGEN(splits_path=TRAIN_SET, tokenizer=_tokenizer, actions_file=actions_file_path)
            
        dataset_train.train = True
        train_dataloader = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

        invpatch_decoder_model = InvPatch(prefix_size=PREFIX_SIZE, config_path=CONFIG_PATH, tokenizer=_tokenizer, 
                        emb_size=DECODER_EMB_SIZE, n_heads=DECODER_HEADS, n_layer=DECODER_LAYERS, T=T, total_patches=TOTAL_PATCHES,
                        pos_enc_h=POS_2D_H, pos_enc_w=POS_2D_W, model_name=METHOD)
        invpatch_decoder_model = invpatch_decoder_model.to(device)

        invpatch_decoder_model.train()

        trained_model = train_decoder(train_model=invpatch_decoder_model, data_loader=train_dataloader, num_epochs=NUM_EPOCHS,
                                    ignore_index=IGNORE_INDEX, learning_rate=LEARNING_RATE, is_multiframe=IS_MULTIFRAME,
                                    smoothing_factor=SMOOTHING_FACTOR, weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS)
        
        # free up space in RAM
        del dataset_train
        del train_dataloader
        
        save_dictionary(f'{WEIGHTS_DIR}/training_params.json', configuration)
        
        torch.save(trained_model.state_dict(), f"{CHECKPOINTS_DIR}/MUGEN_action_gpt.pt")
        trained_model.eval()

        if NUM_FRAMES == 32:
            dataset_test = TestMUGEN_32_Frames(splits_path=TEST_SET, actions_file=actions_file_path)
        else:
            dataset_test = TestMUGEN(splits_path=TEST_SET, actions_file=actions_file_path)
        
        test_eval_dataloader = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        evaluation_response = evaluate(evaluation_model=trained_model, data_loader=test_eval_dataloader, is_multiframe=IS_MULTIFRAME,
                                        verbose=False, temperature=TEMPERATURE)
        save_dictionary(f'{WEIGHTS_DIR}/evaluation_metrics.json', evaluation_response)
        del dataset_test
        del test_eval_dataloader
