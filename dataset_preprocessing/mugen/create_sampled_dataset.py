import json
from pathlib import Path
from tqdm import tqdm
import random
import random

save_path = Path("dataset.json")
save_path.parent.mkdir(exist_ok=True, parents=True)
video_data_path = Path("PATH_TO_RENDERED_VIDEOS")
json_base_path = Path("PATH_TO_JSON_FILES")
videos_to_sample = 16000
video_action_files = []

for video_path in tqdm(video_data_path.glob("*")):
    # videos have "_constructed" appended to the end of their name, remove this part
    json_file_name = f"{video_path.with_suffix('').name[:-12]}.json"
    json_file_path = json_base_path / json_file_name
    
    video_action_files.append({
        "json_file_path": str(json_file_path),
        "video_file_path": str(video_path)
    })
selected_files = random.sample(video_action_files, videos_to_sample)

with open(save_path, "w") as fp:
    json.dump(video_action_files, fp)
