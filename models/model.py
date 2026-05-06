import torch
from transformers import GPT2LMHeadModel
import pickle
import torch.nn as nn
from typing import Tuple, Any
from string import punctuation
import torch.nn.functional as F
from merging_vit import CustomViT
from merging_blocks import StackedBiLSTMClassifier, HANClassifier, MultiframeIntegrationTransformer, InvPatchAggregator, CFAIFAAggregator
from projection_blocks import MLP, ResMLP, MeanResMLP, MeanMLP, MeanEverything, ReshapeMean, MaxpoolLayer, GeM, AttentionPool, GRUAggregator
from vlad_layers import NetVLADLayer

# Our code is based on https://github.com/dhg-wei/DeCap/blob/main/train.py
# We implement changes for additional model configurability, insert a custom tokenizer, and our Global Embedding Module
class GPT2Decoder(nn.Module):
    def __init__(self, prefix_size: int=512, config_path: str = None, bos_id: int=0,
                 eos_id: int=-1, vocab_size: int=19, emb_size=768, n_heads=4, n_layer=4,
                T=16, pos_enc_h=4, pos_enc_w=4, total_patches=49, model_name="MLP"):
        super(GPT2Decoder, self).__init__()
        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        config.bos_token_id = bos_id
        config.eos_token_id = eos_id
        config.vocab_size = vocab_size

        config.n_embd = emb_size
        config.n_head = n_heads
        config.n_layer = n_layer
        config.output_attentions=False

        self.decoder = GPT2LMHeadModel(config)
        self.prefix_size = prefix_size
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.model_name = model_name
        
        if model_name == "Identity":
            self.clip_project = nn.Identity()
        elif model_name == "ReshapeMean":
            self.clip_project = ReshapeMean(T=T)
        elif model_name == "IdentityMean":
            self.clip_project = MeanEverything()
        elif model_name == "Maxpool":
            self.clip_project = MaxpoolLayer()
        elif model_name == "GeM":
            self.clip_project = GeM()
        elif model_name == "MLP":
            self.clip_project = MLP((prefix_size, self.embedding_size))
        elif model_name == "MeanMLP":
            self.clip_project = MeanMLP((prefix_size, self.embedding_size))
        elif model_name == "ResMLP":
            self.clip_project = ResMLP((prefix_size, self.embedding_size))
        elif model_name == "MeanResMLP":
            self.clip_project = MeanResMLP((prefix_size, self.embedding_size))
        elif model_name == "LSTMFrame":
            self.clip_project = StackedBiLSTMClassifier(input_dim=prefix_size, hidden_dim=2048, output_dim=self.embedding_size, num_layers=2)
        elif model_name == "LSTMPatch":
            self.clip_project = GRUAggregator(d_model=prefix_size, T=T)
        elif model_name == "AttentionPool":
            self.clip_project = AttentionPool(dim=prefix_size)
        elif model_name == "Transformer1DFrame":
            self.clip_project = MultiframeIntegrationTransformer(T=16, embed_dim=prefix_size, out_dim=self.embedding_size, layers=3)
        elif model_name == "NetVLADPatch":
            self.clip_project = NetVLADLayer(cluster_num=8, alpha=50.0, T=T, dim=prefix_size, 
                                             num_mlp_layers=1, mlp_hidden_dim=prefix_size, mlp_output_dim=prefix_size)
        elif model_name == "NetVLADFrame":
            self.clip_project = NetVLADLayer(cluster_num=8, alpha=50.0, T=1, dim=prefix_size, 
                                             num_mlp_layers=1, mlp_hidden_dim=prefix_size, mlp_output_dim=prefix_size)
        elif model_name == "Transformer2D":
            self.clip_project = CustomViT(
                                dim=prefix_size,
                                depth=6,
                                heads=int(prefix_size / 192),
                                dim_head=192,
                                mlp_dim=prefix_size,
                                out_dim=self.embedding_size
                            )
        elif model_name == "CFAIFAAggregator":
            self.clip_project = CFAIFAAggregator(width=prefix_size,
                                                    layers=1,
                                                    heads=4,
                                                    output_dim=prefix_size,
                                                    droppath=0.1,
                                                    T=T,
                                                    total_patches=total_patches,
                                                    h=pos_enc_h,
                                                    w=pos_enc_w
                                                )
        elif model_name == "InvPatchAggregator":
            self.clip_project = InvPatchAggregator(patch_merge_input_dim=prefix_size, patch_merge_output_dim=prefix_size, T=T,
                 patches_input_dim=prefix_size, patches_output_dim=prefix_size,
                 in_dim_frames=prefix_size, mlp_dim_frames=prefix_size, total_patches=total_patches,
                 out_dim_frames=self.embedding_size, pos_h=pos_enc_h, pos_w=pos_enc_w)
        else:
            print("Model not registered!")
    
    def forward(self, prefix_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(prefix_features)
        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat, output_attentions=False)
        return out, embedding_clip
    
    def forward_multiframe(self, prefix_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        prefix_projections = self.clip_project(prefix_features)
        
        embedding_cat = torch.cat([prefix_projections, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat, output_attentions=False)
        return out, prefix_projections


class InvPatch(nn.Module):
    def __init__(self, prefix_size: int=512, config_path: str=None, tokenizer: Any=None, 
                    emb_size=768, n_heads=4, n_layer=4, T=16, pos_enc_h=4, pos_enc_w=4, total_patches=49, model_name="MLP"):
        super(InvPatch, self).__init__()

        self._tokenizer = tokenizer
        
        self.decoder = GPT2Decoder(prefix_size=prefix_size, config_path=config_path,
                                   bos_id=self._tokenizer.bos_token_id, eos_id=self._tokenizer.eos_token_id,
                                   vocab_size=self._tokenizer.vocab_size, emb_size=emb_size, n_heads=n_heads, total_patches=total_patches,
                                   n_layer=n_layer, T=T, pos_enc_h=pos_enc_h, pos_enc_w=pos_enc_w, model_name=model_name)

    def __call__(self, prefix_features, gpt_tokens):
        prefix_features = prefix_features.float()
        return self.decoder(prefix_features, gpt_tokens)

    def forward_multiframe(self, prefix_features, gpt_tokens):
        prefix_features = prefix_features.float()
        return self.decoder.forward_multiframe(prefix_features, gpt_tokens)
    
    def decode(self, prefix_features, max_tokens: int = 30, temperature: float = 0.1):
        prefix_features = prefix_features.float()
        embedding_cat = self.decoder.clip_project(prefix_features)
        if len(embedding_cat.shape) < 3:
            embedding_cat = embedding_cat.reshape(-1, 1, prefix_features.shape[-1])
        tokens = None
        for i in range(max_tokens):
            outputs = self.decoder.decoder(inputs_embeds=embedding_cat, output_attentions=False)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = torch.nn.functional.softmax(logits, dim=1)
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = self.decoder.decoder.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == self._tokenizer.eos_token_id:
                break
            embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
        try:
            output_list = list(tokens.squeeze().cpu().numpy())
            output = self._tokenizer.decode(output_list)
        except:
            output = 'None'
        return self.__format_text(output)

    def __format_text(self, generated_text):
        generated_text = generated_text.replace('<BOS>', '').replace('<EOS>', '')
        generated_text = generated_text.replace(" , ", ",").replace(", ", ",")
        generated_text = generated_text.strip(punctuation)
        generated_text = generated_text.strip()
        return generated_text