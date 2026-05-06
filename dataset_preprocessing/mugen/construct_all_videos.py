import subprocess
from pathlib import Path


if __name__ == "__main__":
    data_root = "PATH_TO_DATASET"
    output_folder = "PATH_TO_OUTPUT_FOLDER"
    for run_folder in Path(data_root).glob("*"):
        
        folder_name = run_folder.name
        
        run_json_files = run_folder / "video_metadata"
        for json_file in run_json_files.glob("*.json"):
            conversion_command = f"python construct_data_from_json.py --restore_id {folder_name} --input_json {json_file} --output_folder {output_folder} --save_as_video --frame_id -1 --level_id 1 --gen_original"
            subprocess.run(conversion_command, shell=True)

