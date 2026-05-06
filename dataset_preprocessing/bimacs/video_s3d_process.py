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
from torchvision.models.video import s3d, S3D_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

weights = S3D_Weights.DEFAULT
model = s3d(weights=weights)
model.classifier = torch.nn.Identity().to(device)
model = model.to(device)
model.eval()

preprocess = weights.transforms()

input_data_path = Path("PATH_TO_GIFS")

output_path = Path("PATH_TO_LATENTS_DIR_POOLED")
output_path.mkdir(exist_ok=True, parents=True)

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
                    save_filepath.mkdir(exist_ok=True, parents=True)
                    save_filename = save_filepath / file.with_suffix(".npy").name
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gif = Image.open(file)
                        frames = []
                        try:
                            while True:
                                frame = np.array(gif.convert("RGB"))
                                frame = np.transpose(frame, (2, 1, 0))
                                frames.append(frame)
                                gif.seek(gif.tell() + 1)
                        except EOFError:
                            pass
                        if len(frames) != 16:
                            tqdm.write("Gif has less than 16 frames, skipped.")
                            continue
                        frames = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
                        inputs = preprocess(frames).to(device)
                    try:
                        with torch.no_grad():
                            outputs = model(inputs)
                        latent = outputs.cpu().numpy()
                        np.save(save_filename, latent)
                    except:
                        tqdm.write(f"Conversion failed for: {file}")
                elif file_suffix == ".csv":
                    file_name = file.name
                    copy_to_path = output_path / subject_dir_name / task_dir_name / take_dir_name
                    copy_to_path.mkdir(exist_ok=True, parents=True)
                    shutil.copy(file, copy_to_path)
            

