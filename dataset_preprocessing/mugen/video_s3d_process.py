import os
import warnings
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from torchvision.models.video import s3d, S3D_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = "LATENTS_BASE_PATH"
gif_path = f"PATH_TO_GIFS"
output_directory = f"{base_dir}/latents_video_s3d"

weights = S3D_Weights.DEFAULT
model = s3d(weights=weights)
model.classifier = torch.nn.Identity().to(device)
model = model.to(device)
model.eval()

preprocess = weights.transforms()

skip_existing = False

Path(output_directory).mkdir(exist_ok=True, parents=True)

for idx, gif_path in tqdm(enumerate(Path(gif_path).glob('*.gif')), desc=f"Creating latents..."):
    save_filename = f"{output_directory}/{gif_path.with_suffix('.npy').name}"
    if os.path.isfile(save_filename) and skip_existing:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gif = Image.open(gif_path)
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
            tqdm.write("Gif has less than 16 frames, skipper.")
            continue
        frames = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
        inputs = preprocess(frames).to(device)
    try:
        with torch.no_grad():
            outputs = model(inputs)
        latent = outputs.cpu().numpy()
        np.save(save_filename, latent)
    except:
        tqdm.write(f"Conversion failed for: {gif_path}")