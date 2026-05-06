import av
import numpy as np
import cv2
from PIL import Image
import json
import imageio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from pygifsicle import optimize, gifsicle
from itertools import chain


json_dataset_path = "dataset.json"
gif_dataset_file_save_path = "gif_dataset.json"
save_folder = Path("OUTPUT_PATH_GIFS")
save_folder.mkdir(exist_ok=True, parents=True)

def split_and_skip(lst, chunk_size=16):
    filtered_list = lst[::]  # Skip every second number
    return [filtered_list[i:i + chunk_size] for i in range(0, len(filtered_list), chunk_size)]

def create_windows(array, window=16, stride=12):  # Window len = L, Stride len/stepsize = S
    nrows = ((array.size - window) // stride) + 1
    return array[stride * np.arange(nrows)[:, None] + np.arange(window)]

def create_samples(n, window_size=16, min_overlap=1):
    stride = window_size - min_overlap
    samples = []

    start = 0
    while start + window_size <= n:
        samples.append(list(range(start, start + window_size)))
        start += stride

    # Handle last window if not already included
    if samples and samples[-1][-1] < n - 1:
        samples.append(list(range(n - window_size, n)))

    return samples

def uniform_sample(total_frames: int, num_samples: int) -> list:
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]

def sample_frames(video_path, json_path, overlap=15, num_frames=16, total_frames=96):
    
    sampled_frames = []

    container = av.open(video_path)
    container.seek(0)
    all_frames = np.array(list(container.decode(video=0)))
    container.close()
    indices = split_and_skip(list(range(len(all_frames))))
    indices = create_windows(np.array(list(range(len(all_frames)))), window=num_frames, stride=overlap)
    indices = create_samples(total_frames)
    
    # indices = [uniform_sample(total_frames, 16)]
    create_samples
    
    gif_objects = []
    
    for idx, indice_group in enumerate(indices):

        output_gif_name = save_folder /  f"{Path(video_path).with_suffix('').name}____{idx}.gif"
        
        sampled_frames = all_frames[indice_group]
        sampled_frames = np.stack([frame.to_ndarray(format="rgb24") for frame in sampled_frames])
    
        if len(sampled_frames) == 16:          
            # imageio.mimsave(output_gif_name, sampled_frames, duration=0.1)

            if sampled_frames.dtype != np.uint8:
                sampled_frames = (255 * (sampled_frames - sampled_frames.min()) / (sampled_frames.ptp() + 1e-8)).astype(np.uint8)

            frames = []
            for frame in sampled_frames:
                img = Image.fromarray(frame)
                w, h = img.size
                img = img.resize((w // 2, h // 2), resample=Image.NEAREST)
                frames.append(img)
            frames[0].save(output_gif_name, save_all=True, append_images=frames[1:], duration=0.1, loop=0)

            gifsicle(
                sources=[str(output_gif_name)], # or a single_file.gif
                optimize=True, # Whether to add the optimize flag or not
                colors=256,
                options=['--scale', ' 0.9']
            )

            gif_objects.append({
                'video': str(video_path),
                'json_file': str(json_path),
                'gif_name': str(output_gif_name),
                'sampled_frame_ids': list(indice_group)
            })
            
        else:
            raise Exception(f"Broke for {video_path}")
    return gif_objects


with open(json_dataset_path) as fp:
    dataset = json.load(fp)

gif_dataset = []

def job_function(video_file_path, json_file_path):
    try:
        gif_object = sample_frames(video_file_path, json_file_path)
        return gif_object
    except:
        print("Failed")
        return []

results = Parallel(n_jobs=3)(
            delayed(job_function)(data_object['video_file_path'], data_object['json_file_path']) for data_object in tqdm(dataset))
gif_dataset = list(chain.from_iterable(results))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open(gif_dataset_file_save_path, "w") as fp:
    json.dump(gif_dataset, fp, cls=NpEncoder)