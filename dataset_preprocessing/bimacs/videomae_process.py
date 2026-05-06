from pathlib import Path
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm
import shutil
from transformers import AutoImageProcessor, VideoMAEModel
import numpy as np
import torch
import random
import warnings

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

input_data_path = Path("PATH_TO_GIFS")

output_path = Path("PATH_TO_LATENTS_DIR_POOLED")
output_path_patches = Path("PATH_TO_LATENTS_DIR_PATCHES")
output_path.mkdir(exist_ok=True, parents=True)
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
                    save_filepath = output_path / subject_dir_name / task_dir_name / take_dir_name
                    save_filepath_patches = output_path_patches / subject_dir_name / task_dir_name / take_dir_name
                    
                    save_filepath.mkdir(exist_ok=True, parents=True)
                    save_filepath_patches.mkdir(exist_ok=True, parents=True)
                    
                    save_filename = save_filepath / file.with_suffix(".npy").name
                    save_filename_patches = save_filepath_patches / file.with_suffix(".npy").name
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
                        inputs = image_processor(list(frames_array), return_tensors="pt").to(device)
                    try:
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)
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
                        tqdm.write(f"Conversion failed for: {file}")
                elif file_suffix == ".csv":
                    file_name = file.name
                    copy_to_path = output_path / subject_dir_name / task_dir_name / take_dir_name
                    copy_to_path.mkdir(exist_ok=True, parents=True)
                    shutil.copy(file, copy_to_path)
            


