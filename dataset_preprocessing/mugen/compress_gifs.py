from pygifsicle import optimize, gifsicle
from pathlib import Path
from tqdm import tqdm

for gif_path in tqdm(Path("/home/jovyan/MUGEN_coinrun/dataset_sampled/gifs").glob("*.gif")):
    gifsicle(
        sources=[str(gif_path)], # or a single_file.gif
        optimize=True, # Whether to add the optimize flag or not
        colors=256,
        options=['--scale', ' 0.5']
    )