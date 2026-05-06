import json
import os
import pandas as pd
from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

gif_dataset_path = "gif_dataset.json"
latent_dataset_out_path = "PATH_TO_LATENTS_FOLDER/CSV_FILE"
latents_names = "LATENT_FOLDER_NAME"
header_frame_colums = [f"frame_{idx}" for idx in range(15)]
header = ["latent_filename", *header_frame_colums]

with open(gif_dataset_path) as fp:
    gif_dataset = json.load(fp)

gif_actions_for_csv = []
for dataset_object in tqdm(gif_dataset):
    gif_name = dataset_object['gif_name']
    json_file_path = dataset_object['json_file']
    frame_ids = dataset_object['sampled_frame_ids']

    dir_n = Path(gif_name).parts[6].split("_level")[0]
    json_file_path = Path(*[*Path(json_file_path).parts[:5], dir_n, "video_metadata", *Path(json_file_path).parts[5:]])
    json_file_path = str(json_file_path)
    
    latent_name_parts = list(Path(gif_name).parts)
    # as numpy arrays
    latent_name_parts[-1] = str(Path(latent_name_parts[-1]).with_suffix(".npy"))
    # as torch tensors
    # latent_name_parts[-1] = str(Path(latent_name_parts[-1]).with_suffix(".pt"))
    latent_name_parts[-2] = latents_names 
    lantent_name = os.path.join(*latent_name_parts)
    lantent_name = str(lantent_name).replace("storage", "storage-ext")
    
    if not Path(lantent_name).exists:
        continue

    with open(json_file_path) as fp:
        json_video_description = json.load(fp)
    
    frame_tuples = list(pairwise(frame_ids))
    gif_actions_all = []
    for left_frame_id, right_frame_id in frame_tuples:
        agent_left_frame = json_video_description['frames'][left_frame_id]['agent']
        agent_right_frame = json_video_description['frames'][right_frame_id]['agent']
        
        agent_x_left = agent_left_frame['x']
        agent_y_left = agent_left_frame['y']

        agent_x_right = agent_right_frame['x']
        agent_y_right = agent_right_frame['y']

        dx = agent_x_right - agent_x_left
        dy = agent_y_right - agent_y_left

        move_right = 0
        move_left = 0
        jump = 0
        jump_right = 0
        jump_left = 0
        descend = 0
        inaction = 0
        
        if dy > 0.0:
            if dx == 0.0:
                jump = 1
            elif dx > 0.0:
                jump_right = 1
            elif dx < 0.0:
                jump_left = 1
        elif dy < 0.0:
            descend = 1
        elif dx > 0.0:
            move_right = 1
        elif dx < 0.0:
            move_left = 1
        elif dx == 0.0 and dy == 0.0:
            inaction = 1
        
        gif_actions_all.append([inaction, move_right, move_left, jump, jump_right, jump_left, descend])
    
    gif_actions_short = []
    for frame_actions in gif_actions_all:    
        gif_actions_short.append([index for index, value in enumerate(frame_actions) if value])
    gif_actions_short.insert(0, lantent_name)
    gif_actions_for_csv.append(gif_actions_short)
    
gif_actions_df = pd.DataFrame(gif_actions_for_csv, columns=header)
gif_actions_df.to_csv(latent_dataset_out_path, index=False)