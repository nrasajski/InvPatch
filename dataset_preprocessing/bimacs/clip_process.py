from pathlib import Path
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm
import shutil
import numpy as np
import torch
import random
import warnings
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModel

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

input_data_path = Path("PATH_TO_GIFS")

output_path_pooled = Path("PATH_TO_LATENTS_DIR_POOLED")
output_path_pooled.mkdir(exist_ok=True, parents=True)

output_path_patches = Path("PATH_TO_LATENTS_DIR_PATCHES")
output_path_patches.mkdir(exist_ok=True, parents=True)

save_gif = True
save_actions = True

# top level is subject who is performing actions
for subject_dir in input_data_path.iterdir():
    subject_dir_name = subject_dir.name
    # next is the task performed
    for task_dir in subject_dir.iterdir():
        task_dir_name = task_dir.name
        # each task was performed multiple times
        for take_dir in task_dir.iterdir():
            take_dir_name = take_dir.name
            for file in tqdm(take_dir.iterdir(), leave=True, desc=f"{subject_dir_name}_{task_dir_name}_{take_dir_name}"):
                file_suffix = file.suffix
                if file_suffix == ".gif":
                    
                    save_filepath = output_path_pooled / subject_dir_name / task_dir_name / take_dir_name
                    save_filepath.mkdir(exist_ok=True, parents=True)
                    save_filename = save_filepath / file.with_suffix(".npy").name

                    save_filepath_tokens = output_path_patches / subject_dir_name / task_dir_name / take_dir_name
                    save_filepath_tokens.mkdir(exist_ok=True, parents=True)
                    save_filename_tokens = save_filepath_tokens / file.with_suffix(".npy").name
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gif = Image.open(file)
                        frames = []
                        try:
                            while True:
                                frames.append(np.array(gif.convert("RGB")))
                                gif.seek(gif.tell() + 1)
                        except EOFError:
                            pass
                        frames_array = np.stack(frames)
                        inputs = processor(images=frames_array, return_tensors="pt")
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to(device)
                    try:
                        with torch.no_grad():
                            outputs = model(**inputs)
                        latent_tokens = last_hidden_state = outputs.last_hidden_state.cpu().numpy()
                        latent_aggregated = outputs.pooler_output.cpu().numpy()
                        np.save(save_filename, latent_aggregated)
                        np.save(save_filename_tokens, latent_tokens)
                    except Exception as e:
                        tqdm.write(f"Conversion failed for: {file} with error {e}")
            

