# Preparation of BIMACS dataset

## 1) Download

The dataset is partitioned in multiple .zip files.
Download all files and extract.
Full download instructions and links can be found [here](https://bimanual-actions.humanoids.kit.edu/original_dataset).

## 2) Process into gifs
Set input path (downloaded data) and output path (where to save generated gifs).

Run: `python -m process_raw_data`

## 3) Extract latents
Latents can be pre-computed using one of the following backbones:
* [ViT small](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#models): `python -m vit_small_process`
* [CLIP](https://github.com/openai/CLIP): `python -m clip_process`
* [Video Resnet](https://docs.pytorch.org/vision/main/models/video_resnet.html): `python -m video_resnet_process`
* [Video S3D](https://docs.pytorch.org/vision/main/models/video_s3d.html): `python -m video_s3d_process`
* [VideoMAE](https://huggingface.co/docs/transformers/en/model_doc/videomae): `python -m videomae_process`
* [DINOv2](https://github.com/facebookresearch/dinov2): `python -m dinov2_process`

Note: All latent extraction scripts need to be configured with correct paths

## 4) Process labels
Set input path (gif folder) and output path (latents folder)

Run: `python -m merge_action_files`