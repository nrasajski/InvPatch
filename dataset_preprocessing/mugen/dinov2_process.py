import os
import warnings
from pathlib import Path
from PIL import Image
import random

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
base_dir = "LATENT_BASE_PATH"
gif_path = f"PATH_TO_GIFS"
output_directory = f"{base_dir}/latents_dinov2" # mean pooled embeddings
output_directory_patches = f"{base_dir}/latents_dinov2_patches" # patch embeddings
skip_existing = False

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


transform = transforms.Compose([
    # transforms.ToTensor(),
    ToFloatTensorInZeroOne(),
    Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



Path(output_directory).mkdir(exist_ok=True, parents=True)
Path(output_directory_pathes).mkdir(exist_ok=True, parents=True)

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
        if len(frames) != 16:
            tqdm.write("Gif has less than 16 frames, skipper.")
            continue
        frames_array = np.stack(frames)
        gif_tensor = torch.from_numpy(frames_array)
        image_tensor = transform(gif_tensor).to(device)
    try:
        with torch.no_grad():
            outputs = dinov2_vitg14_reg.forward_features(image_tensor)
        latents_patches = outputs['x_norm_patchtokens'].cpu().numpy()
        latent = outputs['x_norm_clstoken'].cpu().numpy()

        latents_patches = np.float16(latents_patches)
        latent = np.float16(latent)

        np.save(save_filename_patches, latents_patches)
        np.save(save_filename, latent)
    except:
        tqdm.write(f"Conversion failed for: {gif_path}")