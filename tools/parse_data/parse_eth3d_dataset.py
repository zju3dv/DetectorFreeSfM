import argparse
import os.path as osp
import os
from shutil import copyfile
import numpy as np
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import cv2
import PIL
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from src.colmap.read_write_model import qvec2rotmat
from src.utils.colmap.read_write_model import read_model

def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized

def parse_args():
    parser = argparse.ArgumentParser()
    # Input:
    parser.add_argument("--data_base_dir", type=str, default='SfM_dataset/ETH3D_source_data')
    parser.add_argument("--aim_scene_names", default=None)
    parser.add_argument("--img_resize", type=int, default=None)
    parser.add_argument("--img_copy_type", default="soft_link", choices=['copy', 'soft_link'])
    parser.add_argument("--triangulation_mode", type=str, default='True')

    # Output:
    parser.add_argument("--output_base_dir", type=str, default="SfM_dataset/eth3d_triangulation_dataset")
    args = parser.parse_args()
    return args

def get_intrinsic_from_colmap_camera(camera, scale=[1,1]):
    model = camera.model
    scale_x, scale_y = scale # W, H
    if model == 'SIMPLE_RADIAL':
        focal = camera.params[0] / scale_x
        x0 = camera.params[1] / scale_x
        y0 = camera.params[2] / scale_y
        intrinsic = np.array([[focal, 0, x0], [0, focal, y0], [0, 0, 1]])
    elif model == 'PINHOLE':
        focal0 = camera.params[0] / scale_x
        focal1 = camera.params[1] / scale_y
        x0 = camera.params[2] / scale_x
        y0 = camera.params[3] / scale_y
        intrinsic = np.array([[focal0, 0, x0], [0, focal1, y0], [0, 0, 1]])
    else:
        raise NotImplementedError

    return intrinsic

if __name__ == '__main__':
    args = parse_args()

    data_base_dir = args.data_base_dir
    output_base_dir = args.output_base_dir
    aim_scene_names = args.aim_scene_names.split('-') if args.aim_scene_names is not None else None

    scene_names = os.listdir(data_base_dir)
    for scene_name in tqdm(scene_names):
        logger.info(f"Process scene {scene_name}")
        if aim_scene_names is not None:
            if scene_name not in aim_scene_names:
                logger.info(f"Scene:{scene_name} skipped since not in aim scene names")
                continue
        
        scene_path = osp.join(data_base_dir, scene_name)
        try:
            dir_names = os.listdir(scene_path)
        except:
            logger.info(f"Scene:{scene_name} skipped")
            continue

        if args.triangulation_mode != 'False':
            if 'dslr_scan_eval' not in dir_names:
                logger.info(f"No GT scan, skip when parsing Triangulation Dataset")
                continue

        if 'images' not in dir_names or 'dslr_calibration_undistorted' not in dir_names:
            logger.warning(f"Scene:{scene_name} skipped since `images` or `dslr_calibration_undistorted` not in scene directory")
            continue

        # Make output directory:
        output_scene_dir = osp.join(output_base_dir, scene_name)
        if osp.exists(output_scene_dir):
            os.system(f"rm -rf {output_scene_dir}")
        Path(output_scene_dir).mkdir(parents=True)
        images_dir = osp.join(output_scene_dir, 'images')
        poses_dir = osp.join(output_scene_dir, "poses")
        intrins_dir = osp.join(output_scene_dir, 'intrins')
        Path(images_dir).mkdir()
        Path(poses_dir).mkdir()
        Path(intrins_dir).mkdir()

        if args.triangulation_mode != 'False':
            os.system(f"cp -r {osp.join(scene_path, 'dslr_scan_eval')} {osp.join(output_scene_dir, 'dslr_scan_eval')}")

        # Copy images:
        img_rescale_dict = {}
        source_img_dir = osp.join(scene_path, 'images', 'dslr_images_undistorted')
        for img_name in os.listdir(source_img_dir):
            interpolation = "cv2_area"
            if args.img_resize is not None:
                # Resize the long edge to the aim length
                image = Image.open(osp.join(source_img_dir, img_name))
                size = image.size # W * H
                scale = args.img_resize / max(image.size)
                size_new = tuple(int(round(x*scale)) for x in size) # W*H
                image_resized = resize_image(np.asarray(image), size_new, interp=interpolation)

                PIL.Image.fromarray(image_resized.astype(np.uint8)).save(osp.join(images_dir, img_name))

                img_rescale_dict[img_name] = [size[0] / size_new[0], size[1] / size_new[1]]
            else:
                img_rescale_dict[img_name] = [1., 1.]
                if args.img_copy_type == 'copy':
                    copyfile(osp.join(source_img_dir, img_name), osp.join(images_dir, img_name))
                elif args.img_copy_type == 'soft_link':
                    os.system(f"ln -s {osp.join(source_img_dir, img_name)} {osp.join(images_dir, img_name)}")
                else:
                    raise NotImplementedError

        # Parse camera pose and instrinsic from colmap-like format:
        model_path = osp.join(scene_path, 'dslr_calibration_undistorted')
        cameras, images, points3D = read_model(model_path)
        for image_id, image in images.items():
            image_name = osp.basename(image.name)
            image_base_name = osp.splitext(image_name)[0]

            # Parse pose:
            r = qvec2rotmat(image.qvec)
            t = image.tvec
            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = t
            np.savetxt(osp.join(poses_dir, image_base_name + '.txt'), T)

            # Parse intrin:
            camera_id = image.camera_id
            camera = cameras[camera_id]
            scale = img_rescale_dict[image_name]
            intrin = get_intrinsic_from_colmap_camera(camera, scale=scale)

            np.savetxt(osp.join(intrins_dir, image_base_name + '.txt'), intrin)