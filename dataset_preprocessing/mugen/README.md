# Preparation of MUGEN dataset

Code was developed by extending the official dataset [repository](https://github.com/mugen-org/MUGEN_coinrun)

## 1) Download
Download and extract the full dataset.
Download instructions and links can be found [here](https://mugen-org.github.io/download).

## 2) Render videos
The dataset is composed of json files that describe videos. Videos need to be rendered.
The code for rendering videos can be found [here](https://github.com/mugen-org/MUGEN_coinrun). 

Either clone the repository into this folder or copy this folder’s contents into the cloned repository.

To render videos set paths to input (extracted dataset) and output (location where rendered videos will be stored) and run the following script:

`python -m construct_all_videos`

## 2) Process into gifs
Randomly sample desired number of videos from whole dataset by setting all parameters and running:

`python -m create_sampled_dataset`

Set input path (downloaded data) and output path (where to save generated gifs)

Run: `python -m create_gifs`

## 3) Extract latents
Latents can be pre-computed using one of the following backbones:
* [Video Resnet](https://docs.pytorch.org/vision/main/models/video_resnet.html): `python -m video_resnet_process`
* [Video S3D](https://docs.pytorch.org/vision/main/models/video_s3d.html): `python -m video_s3d_process`
* [VideoMAE](https://huggingface.co/docs/transformers/en/model_doc/videomae): `python -m videomae_process`
* [DINOv2](https://github.com/facebookresearch/dinov2): `python -m dinov2_process`

Note: All latent extraction scripts need to be configured with correct paths

## 4) Process labels
Set input path (gif folder) and output path (latents folder)

Run: `python -m create_gif_actions`