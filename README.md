# Detector-Free Structure from Motion
### [Project Page](https://zju3dv.github.io/DetectorFreeSfM/) | [Paper](https://zju3dv.github.io/DetectorFreeSfM/files/main_paper_with_sup.pdf)
<br/>

> Detector-Free Structure from Motion                                                                                                                                                
> [Xingyi He](https://github.com/hxy-123/), [Jiaming Sun](https://jiamingsun.ml), [Yifan Wang](https://github.com/wyf2020), [Sida Peng](https://pengsida.net/), [Qixing Huang](https://www.cs.utexas.edu/~huangqx/), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Xiaowei Zhou](https://xzhou.me)                              
> CVPR 2024, [1st](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407) in Image Matching Challenge 2023

![demo_vid](assets/demo.gif)

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation instructions.

## Prepare Dataset
The data structure of our system is organized as follows:
```
repo_path/SfM_dataset
    - dataset_name1
        - scene_name_1
            - images
                - image_name_1.jpg or .png or ...
                - image_name_2.jpg
                - ...
            - intrins (optional, used for evaluation)
                - camera_name_1.txt
                - camera_name_2.txt
                - ...
            - poses (optional, used for evaluation)
                - pose_name_1.txt
                - pose_name_2.txt
                - ...
        - scene_name_2
            - ...
    - dataset_name2
        - ...
```
The folder naming of `images`, `intrins` and `poses` is **compulsory**, for the identification by our system.

Now, download the training and evaluation datasets, and then format them to required structure following instructions in [DATASET_PREPARE.md](DATASET_PREPARE.md).

## Run Demo data
First modify L22 in `hydra_configs/demo/dfsfm.yaml` to specify the absolute path of the repo.
Then run the following command:
```
python eval_dataset.py +demo=dfsfm.yaml
```
SfM result will be saved in `SfM_dataset/example_dataset/example_scene/DetectorFreeSfM_loftr_official_coarse_only__scratch_no_intrin/colmap_refined` in COLMAP format, and can be visualized by `colmap gui`.

## Evaluation
### SfM Evaluation
```
# For ETH3D dataset:
python eval_dataset.py +eth3d_sfm=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='loftr_official'
python eval_dataset.py +eth3d_sfm=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='aspanformer'
python eval_dataset.py +eth3d_sfm=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='matchformer'

# For IMC dataset:
sh scripts/eval_imc_dataset.sh

# For TexturePoorSfM dataset:
sh scripts/eval_texturepoorsfm_dataset.sh
```
### Triangulation evaluation

```
# For ETH3D dataset:
python eval_dataset.py +eth3d_tri=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='loftr_official'
python eval_dataset.py +eth3d_tri=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='aspanformer'
python eval_dataset.py +eth3d_tri=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='matchformer'
```

### Tips about speed up
1. You can speed up evaluation by enable multi-processing if you have multiple GPUs.
You can set `ray.enable=True` and set `ray.n_workers=your_gpu_number` the configs to simutaneously evaluate many scenes within a dataset.
2. For a scene with many images, like `Bridge` in the ETH3D dataset, you can set multiple workers for image matching in coarse SfM and multi-view refinement matching phase by setting `sub_use_ray=True` and `sub_ray_n_worker=your_gpu_number`
3. Increase batchsize in multi-view refinement phase. Currently, we chunk the tracks in refinement matching and `neuralsfm.NEUSFM_refinement_chunk_size` is set to `2000` so that it can work on GPU with VRAM less than 12GB. If the GPUs in your device are with larger VRAM, you can consider increase this value to speed up the process.

## Train Multiview Matching Refiner
Be sure you have downloaded and formated the MegaDepth dataset following [DATASET_PREPARE.md](DATASET_PREPARE.md).
```
python train_multiview_matcher.py +experiment=multiview_refinement_matching.yaml paths=dataset_path_config trainer=trainer_config
```
You can modify the GPU ids in `hydra_training_configs/trainer/trainer_config.yaml`. By default, we use 8 GPUs for training.

## Acknowledgement
Our code is partally based on [COLMAP](https://github.com/colmap/colmap) and [HLoc](https://github.com/cvg/Hierarchical-Localization), we thank the authors for their great work.

## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```
@article{he2024dfsfm,
  title={Detector-Free Structure from Motion},
  author={He, Xingyi and Sun, Jiaming and Wang, Yifan and Peng, Sida and Huang, Qixing and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2024}
}
```