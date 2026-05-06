from pathlib import Path
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm
import subprocess
from pygifsicle import optimize, gifsicle
import os

input_data_path = Path(r"PATH_TO_DOWNLOADED_DATASET")
input_labels_path = Path(r"PATH_TO_LABELS")

output_path = Path(r"SAVE_PATH")
output_path.mkdir(exist_ok=True, parents=True)

image_extension = "webp"
save_gif = True
save_actions = True


def create_output_header_per_frame():
    header = ["gif_filename"]
    for i in range(16):
        header.append(f"frame_{i}")
    return header

def split_and_skip(lst, chunk_size=16):
    filtered_list = lst[::]  # Skip every second number
    return [filtered_list[i:i + chunk_size] for i in range(0, len(filtered_list), chunk_size)]


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

# top level is subject who is performing actions
for subject_dir in input_data_path.iterdir():
    subject_dir_name = subject_dir.name
    # next is the task performed
    for task_dir in subject_dir.iterdir():
        task_dir_name = task_dir.name
        # each task was performed multiple times
        for take_dir in task_dir.iterdir():
            take_dir_name = take_dir.name
            # dictionary that holds all frames because order is important
            images = {}
            # use only rbg frames and ignore the rest
            rbg_dir = "rgb"
            # each video of a performed task is chunked into multiple folders of frames
            for chunk_dir in (take_dir / rbg_dir).iterdir():
                # check if it's a directory because metadata file is also in here somewhere
                if chunk_dir.is_dir():
                    for image in chunk_dir.glob(f"*.{image_extension}"):
                        # extract frame number and use that as key in dictionary
                        image_name = image.with_suffix("").name
                        frame_num = image_name.split("_")[1]
                        images[frame_num] = image

            # load and process labels
            labels_filepath = (input_labels_path /  subject_dir_name / task_dir_name / take_dir_name).with_suffix(".json")
            with open(labels_filepath, "r") as f:
                labels = json.load(f)

            right_hand = labels['right_hand']
            right_hand_ranges = [value for idx, value in enumerate(right_hand) if idx % 2 == 0]
            right_hand_action_ids = [value for idx, value in enumerate(right_hand) if idx % 2 == 1]
            right_hand_processed = [
                {"range": (right_hand_ranges[i], right_hand_ranges[i + 1]), "value": right_hand_action_ids[i]}
                for i in range(len(right_hand_action_ids))
            ]

            left_hand = labels['left_hand']
            left_hand_ranges = [value for idx, value in enumerate(left_hand) if idx % 2 == 0]
            left_hand_action_ids = [value for idx, value in enumerate(left_hand) if idx % 2 == 1]
            left_hand_processed = [
                {"range": (left_hand_ranges[i], left_hand_ranges[i + 1]), "value": left_hand_action_ids[i]}
                for i in range(len(left_hand_action_ids))
            ]

            # divide all frames into gifs, skips every second frame
            frame_ids = list(range(len(images.keys())))
            sampled_gif_frame_ids = create_samples(len(frame_ids))
            sampled_gif_frame_ids = [sublist for sublist in sampled_gif_frame_ids if len(sublist) == 16]

            gif_save_path = output_path / subject_dir_name / task_dir_name / take_dir_name
            gif_save_path.mkdir(exist_ok=True, parents=True)

            gif_frame_descriptions = {}
            all_gif_actions = []
            for idx, gif_frame_ids in tqdm(enumerate(sampled_gif_frame_ids), leave=True, desc=f"{subject_dir_name}_{task_dir_name}_{take_dir_name}"):
                gif_name = f"gif_{idx}.gif"
                gif_frame_descriptions[gif_name] = gif_frame_ids
                frame_list = []
                gif_actions = [gif_name]
                for frame_id in gif_frame_ids:
                    frame_path = images[f"{frame_id}"]

                    frame = Image.open(frame_path)
                    frame_list.append(frame)

                    # process actions for these frames
                    frame_actions = []
                    for actions_object in left_hand_processed:
                        if actions_object["range"][0] <= frame_id <= actions_object["range"][1]:
                            frame_actions.append(actions_object["value"])
                            break
                    for actions_object in right_hand_processed:
                        if actions_object["range"][0] <= frame_id <= actions_object["range"][1]:
                            frame_actions.append(actions_object["value"])
                            break
                    if len(frame_actions) == 0:
                        print(left_hand_processed)
                        print(right_hand_processed)
                        print(subject_dir_name, task_dir_name, take_dir_name, frame_id)
                        break
                    gif_actions.append(frame_actions)

                if save_gif:
                    frame_list[0].save(
                        gif_save_path / gif_name,
                        save_all=True,
                        append_images=frame_list[1:],
                        duration=100,
                        loop=0
                    )
                    gifsicle(
                        sources=[str(gif_save_path / gif_name)], # or a single_file.gif
                        optimize=True, # Whether to add the optimize flag or not
                        colors=256,
                        options=['--scale', ' 0.9']
                    )
                    
                all_gif_actions.append(gif_actions)
            if save_actions:
                gif_dataset_df = pd.DataFrame(all_gif_actions, columns=create_output_header_per_frame())
                gif_dataset_df.to_csv(gif_save_path / "gif_actions.csv", index=False)
            with open(gif_save_path / "gif_descriptions.json", "w") as f:
                json.dump(gif_frame_descriptions, f)

