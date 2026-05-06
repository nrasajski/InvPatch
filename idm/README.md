# Overview
All models (Global Embedding Modules, Action Decoder etc.) are in the ``models`` folder.

Each dataset has a dedicated training folder containing additional custom dataset-specific logic and stored training artifacts.
* MUGEN: ``idm_mugen``
* BIMACS: ``idm_bimacs``

Despite these differences, all follow the same training pipeline.

# Training

## 0) Prepare environment and data
All necessary libraries are specified in ``requirements.txt`` file in project root

Data should be downloaded and pre-processed according to instructions in ``dataset_preprocessing``.

Open a terminal window and position to ``PROJECT_ROOT/InvPatch/idm``

## 1) Create train and test splits
To create train-test splits set path to processed data, split ratio, save path and execute:

`python -m IDM_FOLDER.split_data`

## 2) Prepare experiment configurations
File ``exp_configurations.py`` holds the definitions of all experimental configurations. 
Each configuration is a python dictionary of hyperparameters and model configurations. Configurations can be grouped into arrays.

## 3) Start experiments
In file ``train_and_eval_idm.py`` set ``PROJECT_ROOT``.
To train and evaluate trained IDM run:

`python -m IDM_FOLDER.train_and_eval_idm`
