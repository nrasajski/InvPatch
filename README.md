# InvPatch
This project contains the code used in the paper InvPatch: Prefix-Based Conditional Generation for Inverse Dynamics. 
The experiments were conducted on a Linux system. The total size of all datasets when pre-processed is aroud 2TB.
Experiments were run on a single A6000 GPU.


## Project structure
All the code for downloading and pre-processing the datasets 
used in the paper is contained within the ``dataset_preprocessing`` folder.
For detailed instructions, refer to the README file for each dataset.
 
The IDM code is located in the ``idm`` folder, with detailed instructions in its README file.

## Getting started

All necessary libraries are specified in ``requirements.txt`` file in root of this project. 

## Maintenance

This repo shares code used for academic research, it's not production ready (robust across operating systems, python versions, python packages etc.). There's no plan to actively maintain this repo for these purposes, nor to fix minor bugs.
