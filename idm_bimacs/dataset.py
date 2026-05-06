from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
import random
from pathlib import Path
from typing import Any
import json
from tqdm import tqdm
from collections import defaultdict


device = "cuda" if torch.cuda.is_available() else "cpu"


class InvPatchBimacsTrainDataset(Dataset):
    def __init__(self,
                 splits_path,
                 tokenizer,
                 cache_size=14000,
                 actions_file: str = 'gif_actions_stride_16_per_window.csv'):

        self.text_sequence = ["idle", "approach", "retreat", "lift", "place", "hold", "pour", "cut", "hammer", "saw", "stir", "screw", "drink", "wipe"]
        # self.text_sequence_r = ["r_idle", "r_approach", "r_retreat", "r_lift", "r_place", "r_hold", "r_pour", "r_cut", "r_hammer", "r_saw", "r_stir", "r_screw", "r_drink", "r_wipe"]
        # self.text_sequence_l = ["l_idle", "l_approach", "l_retreat", "l_lift", "l_place", "l_hold", "l_pour", "l_cut", "l_hammer", "l_saw", "l_stir", "l_screw", "l_drink", "l_wipe"]
        with open(splits_path, 'r') as f:
            self.latents = json.load(f)
        self.actions_file = actions_file
        self.latent_files = []
        self.action_strings = []
        self.tokenizer = tokenizer
        self.cache = {}
        self.cache_size = cache_size

        game_latent_files = [Path(latent_path) for latent_path in self.latents]
        # load the actions CSV
        game_actions_df = pd.read_csv(self.actions_file)
        print(game_actions_df.head()['latent_filename'].tolist())
        self.latent_files.extend(game_latent_files)
        for latent_file in tqdm(game_latent_files, desc="Loading latents and actions"):         
            if len(self.cache) < self.cache_size:
                self.cache[latent_file] = torch.load(latent_file).squeeze()
                
            action_row = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file)].iloc[:, :]
            actions = action_row.values[0, 1:]
            # actions = action_row.values[0, 1:-1]
            actions_array = []
            for frame_actions in actions:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])


            action_string = ",".join(actions_array) if actions_array else "inaction"
            self.action_strings.append(action_string)

        self.unique_actions = list(set(self.action_strings))
        self.action_categorical_mapping = {string: index for index, string in enumerate(self.unique_actions)}
        self.custom_tokens_dict = {}

        for unique_action_string in tqdm(self.unique_actions, desc='Extracting text tokens'):
            tokenized_action_custom = self.tokenizer.encode(unique_action_string.lower())[0]
            self.custom_tokens_dict[unique_action_string] = tokenized_action_custom

        print("Dataset loaded")

    def __len__(self):
        return len(self.latent_files)


    def __getitem__(self, idx):
        action_string = self.action_strings[idx]

        latent_file_path = self.latent_files[idx]
         # Use cache if available
        if latent_file_path in self.cache:
            latent_data = self.cache[latent_file_path]
        else:
            latent_data = torch.load(latent_file_path, map_location="cpu")
            latent_data = latent_data.squeeze()
        
        custom_tokens = self.custom_tokens_dict[action_string]
        return latent_data, custom_tokens, action_string


class InvPatchBimacsTrainDataset_32_Frames(Dataset):
    def __init__(self,
                 splits_path,
                 tokenizer,
                 cache_size=65000,
                 actions_file: str = 'gif_actions_stride_16_per_window.csv'):

        self.text_sequence = ["idle", "approach", "retreat", "lift", "place", "hold", "pour", "cut", "hammer", "saw", "stir", "screw", "drink", "wipe"]
        # self.text_sequence_r = ["r_idle", "r_approach", "r_retreat", "r_lift", "r_place", "r_hold", "r_pour", "r_cut", "r_hammer", "r_saw", "r_stir", "r_screw", "r_drink", "r_wipe"]
        # self.text_sequence_l = ["l_idle", "l_approach", "l_retreat", "l_lift", "l_place", "l_hold", "l_pour", "l_cut", "l_hammer", "l_saw", "l_stir", "l_screw", "l_drink", "l_wipe"]
        with open(splits_path, 'r') as f:
            self.latents = json.load(f)
        self.actions_file = actions_file
        self.latent_files = []
        self.action_strings = []
        self.tokenizer = tokenizer
        self.cache = {}
        self.cache_size = cache_size

        game_latent_files = [Path(latent_path) for latent_path in self.latents]
        # load the actions CSV
        game_actions_df = pd.read_csv(self.actions_file)

        take_groups = defaultdict(list)
        for path in game_latent_files:
            group_key = str(Path(*path.parts[:8]))
            take_groups[group_key].append(path)
        
        pairs_list = []
        for group_key, files in take_groups.items():
            # Extract idx and sort by it
            sorted_files = sorted(files, key=lambda x: int(x.with_suffix("").name.split('_')[-1]))
            filepaths = [str(f) for f in sorted_files]

            # Generate consecutive pairs
            pairs = []
            i = 0
            while i < len(filepaths) - 1:
                pairs.append((filepaths[i], filepaths[i + 1]))
                i += 2
            if len(filepaths) % 2 == 1 and len(filepaths) > 1:
                # Repeat second-to-last in the final pair
                pairs.append((filepaths[-2], filepaths[-1]))
            pairs_list.extend(pairs)

        for idx, pair in tqdm(enumerate(pairs_list), desc="Merging latents"):
            
            latent_file_1, latent_file_2 = pair
            
            if len(self.cache) < self.cache_size:
                latent_1 = torch.load(latent_file_1).squeeze()
                latent_2 = torch.load(latent_file_2).squeeze()
                latent_data = torch.cat([latent_1, latent_2], dim=0)
                self.cache[idx] = latent_data
            
            self.latent_files.append((latent_file_1, latent_file_2))
            
            action_row_1 = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file_1)].iloc[:, :]
            action_row_2 = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file_2)].iloc[:, :]
            
            actions_1 = action_row_1.values[0, 1:]
            actions_2 = action_row_2.values[0, 1:]
            
            actions_array = []
            
            for frame_actions in actions_1:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])

            for frame_actions in actions_2:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])
            
            action_string = ",".join(actions_array) if actions_array else "inaction"
            self.action_strings.append(action_string)

        self.unique_actions = list(set(self.action_strings))
        self.action_categorical_mapping = {string: index for index, string in enumerate(self.unique_actions)}
        self.custom_tokens_dict = {}

        for unique_action_string in tqdm(self.unique_actions, desc='Extracting text tokens'):
            tokenized_action_custom = self.tokenizer.encode(unique_action_string.lower())[0]
            self.custom_tokens_dict[unique_action_string] = tokenized_action_custom

        print("Dataset loaded")

    def __len__(self):
        return len(self.latent_files)


    def __getitem__(self, idx):
        action_string = self.action_strings[idx]
        custom_tokens = self.custom_tokens_dict[action_string]
        
        if idx in self.cache:
            latent_data = self.cache[idx]
        else:    
            latent_file_1, latent_file_2 = self.latent_files[idx]
            latent_data_1 = torch.load(latent_file_1, map_location="cpu").squeeze()
            latent_data_2 = torch.load(latent_file_2, map_location="cpu").squeeze()
            latent_data = torch.cat([latent_data_1, latent_data_2], dim=0)
        
        return latent_data, custom_tokens, action_string


class InvPatchBimacsTestDataset(Dataset):
    def __init__(self,
                 splits_path,                 
                 actions_file: str = 'gif_actions_stride_16_per_window.csv'):

        self.text_sequence = ["idle","approach", "retreat", "lift", "place", "hold", "pour", "cut", "hammer", "saw", "stir", "screw", "drink", "wipe"]
        # self.text_sequence_r = ["r_idle", "r_approach", "r_retreat", "r_lift", "r_place", "r_hold", "r_pour", "r_cut", "r_hammer", "r_saw", "r_stir", "r_screw", "r_drink", "r_wipe"]
        # self.text_sequence_l = ["l_idle", "l_approach", "l_retreat", "l_lift", "l_place", "l_hold", "l_pour", "l_cut", "l_hammer", "l_saw", "l_stir", "l_screw", "l_drink", "l_wipe"]
        with open(splits_path, 'r') as f:
            self.latents = json.load(f)
        self.actions_file = actions_file
        self.game_action_latent_map = {}
        self.game_latent_map = {}
        self.latent_files = []
        self.action_strings = []

        game_latent_files = [Path(latent_path) for latent_path in self.latents]
        # load the actions CSV
        game_actions_df = pd.read_csv(self.actions_file)

        self.latent_files.extend(game_latent_files)
        for latent_file in game_latent_files:
            action_row = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file)].iloc[:, :]

            # actions = action_row.values[0, 1:]
            actions = action_row.values[0, 1:]
            actions_array = []
            for frame_actions in actions:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])
            
            action_string = ",".join(actions_array) if actions_array else "inaction"
            self.action_strings.append(action_string)

        self.unique_actions = list(set(self.action_strings))
        self.action_categorical_mapping = {string: index for index, string in enumerate(self.unique_actions)}
        self.categorical_action_mapping = {value: key for key, value in self.action_categorical_mapping.items()}
        print("Dataset loaded")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        action_string = self.action_strings[idx]

        latent_file = self.latent_files[idx]
        latent_data = torch.load(latent_file).squeeze()

        return latent_data, action_string, str(latent_file)




class InvPatchBimacsTestDataset_32_Frames(Dataset):
    def __init__(self,
                 splits_path,                 
                 actions_file: str = 'gif_actions_stride_16_per_window.csv'):

        self.text_sequence = ["idle","approach", "retreat", "lift", "place", "hold", "pour", "cut", "hammer", "saw", "stir", "screw", "drink", "wipe"]
        # self.text_sequence_r = ["r_idle", "r_approach", "r_retreat", "r_lift", "r_place", "r_hold", "r_pour", "r_cut", "r_hammer", "r_saw", "r_stir", "r_screw", "r_drink", "r_wipe"]
        # self.text_sequence_l = ["l_idle", "l_approach", "l_retreat", "l_lift", "l_place", "l_hold", "l_pour", "l_cut", "l_hammer", "l_saw", "l_stir", "l_screw", "l_drink", "l_wipe"]
        with open(splits_path, 'r') as f:
            self.latents = json.load(f)
        self.actions_file = actions_file
        self.game_action_latent_map = {}
        self.game_latent_map = {}
        self.latent_files = []
        self.action_strings = []

        game_latent_files = [Path(latent_path) for latent_path in self.latents]
        # load the actions CSV
        game_actions_df = pd.read_csv(self.actions_file)

        take_groups = defaultdict(list)
        for path in game_latent_files:
            group_key = str(Path(*path.parts[:8]))
            take_groups[group_key].append(path)
        
        pairs_list = []
        for group_key, files in take_groups.items():
            # Extract idx and sort by it
            sorted_files = sorted(files, key=lambda x: int(x.with_suffix("").name.split('_')[-1]))
            filepaths = [str(f) for f in sorted_files]

            # Generate consecutive pairs
            pairs = []
            i = 0
            while i < len(filepaths) - 1:
                pairs.append((filepaths[i], filepaths[i + 1]))
                i += 2
            if len(filepaths) % 2 == 1 and len(filepaths) > 1:
                # Repeat second-to-last in the final pair
                pairs.append((filepaths[-2], filepaths[-1]))
            pairs_list.extend(pairs)

        for pair in pairs_list:
            
            latent_file_1, latent_file_2 = pair
            self.latent_files.append((latent_file_1, latent_file_2))
            
            action_row_1 = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file_1)].iloc[:, :]
            action_row_2 = game_actions_df.loc[game_actions_df['latent_filename'] == str(latent_file_2)].iloc[:, :]
            
            actions_1 = action_row_1.values[0, 1:]
            actions_2 = action_row_2.values[0, 1:]
            
            actions_array = []
            
            for frame_actions in actions_1:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])

            for frame_actions in actions_2:
                left, right = eval(frame_actions)
                if left is None or left == "None":
                    left = 0
                if right is None or right == "None":
                    right = 0
                # actions_array.append(self.text_sequence_l[left])
                # actions_array.append(self.text_sequence_r[right])
                actions_array.append(self.text_sequence[left])
                actions_array.append(self.text_sequence[right])

            
            action_string = ",".join(actions_array) if actions_array else "inaction"
            self.action_strings.append(action_string)

        self.unique_actions = list(set(self.action_strings))
        self.action_categorical_mapping = {string: index for index, string in enumerate(self.unique_actions)}
        self.categorical_action_mapping = {value: key for key, value in self.action_categorical_mapping.items()}
        print("Dataset loaded")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        action_string = self.action_strings[idx]

        latent_file_1, latent_file_2 = self.latent_files[idx]
        latent_data_1 = torch.load(latent_file_1).squeeze()
        latent_data_2 = torch.load(latent_file_2).squeeze()

        latent_data = torch.cat([latent_data_1, latent_data_2], dim=0)

        return latent_data, action_string