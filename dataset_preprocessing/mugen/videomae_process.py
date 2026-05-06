import os
import random
import warnings
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEModel

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
base_dir = "PATH_TO_LATENTS_BASE_DIR"
gif_path = f"PATH_TO_GIFS"
output_directory = f"{base_dir}/latents_dinov2" # mean pooled embeddings
output_directory_patches = f"{base_dir}/latents_dinov2_patches" # patch embeddings
skip_existing = False

Path(output_directory).mkdir(exist_ok=True, parents=True)

for idx, gif_path in tqdm(enumerate(Path(gif_path).glob('*.gif')), desc=f"Creating latents..."):
    save_filename = f"{output_directory}/{gif_path.with_suffix('.npy').name}"
    save_filename_patches = f"{output_directory_pathes}/{gif_path.with_suffix('.npy').name}"
    if os.path.isfile(save_filename) and skip_existing:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frames.append(np.array(gif.convert("RGB")))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        frames_array = np.stack(frames)
        inputs = image_processor(list(frames_array), return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        latent_patches = last_hidden_states
        latent_patches = latent_patches.cpu().numpy()
        latent_patches = np.float16(latent_patches)

        latent_mean = last_hidden_states.mean(1)
        latent_mean = latent_mean.cpu().numpy()
        latent_mean = np.float16(latent_mean)

        np.save(save_filename, latent)
        np.save(save_filename_patches, latent_patches)
    except:
        tqdm.write(f"Conversion failed for: {gif_path}")