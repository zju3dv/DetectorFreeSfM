import argparse
import os.path as osp
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
import numpy as np
from loguru import logger
from tqdm import tqdm
from src.utils.data_io import load_h5

def parse_args():
    parser = argparse.ArgumentParser()
    # Input:
    parser.add_argument("--IMC_base_dir", type=str, default='SfM_dataset/IMC2021')
    parser.add_argument("--dataset_name", type=str, default='phototourism')
    parser.add_argument("--scene_name", type=str, default='british_museum')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])

    # Output:
    parser.add_argument("--output_base_dir", type=str, default="SfM_dataset/IMC_dataset")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    imc_base_dir = args.IMC_base_dir
    output_base_dir = args.output_base_dir

    scene_path = osp.join(imc_base_dir, args.dataset_name, args.scene_name, 'set_100')
    logger.info(f'Parsing: {scene_path}')

    subset_dir = osp.join(scene_path, 'sub_set')
    subset_names = [subset_name for subset_name in os.listdir(subset_dir) if '._' not in subset_name]

    output_dataset_name = "_".join([args.dataset_name, args.scene_name]) # e.g. phototourism_edinburgh

    for subset_name in tqdm(subset_names):
        if '.txt' not in subset_name or '._' in subset_name:
            continue
        subset_path = osp.join(subset_dir, subset_name)
        output_scene_name = osp.splitext(osp.basename(subset_path))[0] # e.g. 10bag_xxx

        # Output directory:
        output_scene_dir = osp.join(output_base_dir, output_dataset_name, output_scene_name)
        images_dir = osp.join(output_scene_dir, 'images')
        poses_dir = osp.join(output_scene_dir, "poses")
        intrins_dir = osp.join(output_scene_dir, 'intrins')
        if osp.exists(output_scene_dir):
            os.system(f"rm -rf {output_scene_dir}")

        Path(images_dir).mkdir(parents=True, exist_ok=True)
        Path(poses_dir).mkdir(parents=True, exist_ok=True)
        Path(intrins_dir).mkdir(parents=True, exist_ok=True)

        with open(subset_path, "r") as f:
            img_relative_paths = f.read().splitlines()

        img_paths = [osp.join(scene_path, img_relative_path) for img_relative_path in img_relative_paths]

        for img_path in img_paths:
            img_name = osp.basename(img_path)
            img_base_name = osp.splitext(img_name)[0]
            # Load gt pose and intrinsic

            calib_path = osp.join(scene_path, 'calibration', 'calibration_' + img_base_name + '.h5')
            calib = load_h5(calib_path)
            K, R, t = calib['K'], calib['R'], calib['T']

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            # Output:
            os.system(f"ln -s {img_path} {osp.join(images_dir, img_name)}")
            np.savetxt(osp.join(intrins_dir, img_base_name + '.txt'), K) # 3*3
            np.savetxt(osp.join(poses_dir, img_base_name + '.txt'), T) # 4*4