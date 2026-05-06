from pathlib import Path
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm
import shutil

input_data_path = Path("PROCESSED_DATA_PATH")

output_path = Path("LATENTS_FOLDER_PATH")
output_path.mkdir(exist_ok=True, parents=True)

save_gif = True
save_actions = True

action_dfs = []
# top level is subject who is performing actions
for subject_dir in input_data_path.iterdir():
    subject_dir_name = subject_dir.name
    # next is the task performed
    for task_dir in subject_dir.iterdir():
        task_dir_name = task_dir.name
        # each task was performed multiple times
        for take_dir in task_dir.iterdir():
            take_dir_name = take_dir.name
            for actions_file in take_dir.glob("*.csv"):
                action_df = pd.read_csv(actions_file)
                latent_path = output_path / subject_dir_name / task_dir_name / take_dir_name
                action_df["gif_filename"] = action_df["gif_filename"].apply(lambda x: str(latent_path) + "/" + Path(x).with_suffix(".npy").name)
                action_df.rename(columns={"gif_filename": "latent_filename"}, inplace=True)
                action_dfs.append(action_df)

            
# Concatenate all DataFrames into one
merged_df = pd.concat(action_dfs, ignore_index=True)
# Save the merged DataFrame to a new CSV file (optional)
merged_df.to_csv(output_path / "all_actions.csv", index=False)
