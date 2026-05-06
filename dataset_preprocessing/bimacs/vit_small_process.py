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
import timm

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"



input_data_path = Path("PATH_TO_GIFS")

output_path_pooled = Path("PATH_TO_LATENTS_DIR_POOLED")
output_path_pooled.mkdir(exist_ok=True, parents=True)

output_path_patches = Path("PATH_TO_LATENTS_DIR_PATCHES")
output_path_patches.mkdir(exist_ok=True, parents=True)

model = timm.create_model(
    'vit_small_patch32_224.augreg_in21k_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)



def to_normalized_float_tensor(vid):
    return vid.permute(0, 3, 1, 2).to(torch.float32) / 255.0


def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

image_processor = transforms.Compose([
    ToFloatTensorInZeroOne(),
    Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
                        images = image_processor(torch.from_numpy(frames_array).to(device))
                    try:
                        with torch.no_grad():
                            output = model.forward_features(images)
                        latent_tokens = output[:, 1:, :].cpu().numpy()
                        latent_aggregated = output[:, 0, :].cpu().numpy()
                        np.save(save_filename, latent_aggregated)
                        np.save(save_filename_tokens, latent_tokens)
                    except Exception as e:
                        tqdm.write(f"Conversion failed for: {file} with error {e}")
            

