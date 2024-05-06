import cv2
import logging
import os.path as osp
from loguru import logger
import numpy as np
import PIL

from pathlib import Path
from src.utils.colmap.read_write_model import Camera, Image, read_model
from src.utils.colmap.read_write_model import rotmat2qvec
from src.utils.colmap.read_write_model import write_model

def get_pose_from_txt(img_index, pose_dir, is_c2w=True):
    """ Read 4x4 transformation matrix from txt """
    pose_file = osp.join(pose_dir, '{}.txt'.format(img_index))
    pose = np.loadtxt(pose_file)
    if is_c2w:
        # Convert pose to w2c
        pose = np.linalg.inv(pose)
    
    tvec = pose[:3, 3].reshape(3, )
    qvec = rotmat2qvec(pose[:3, :3]).reshape(4, )
    return pose, tvec, qvec


def convert_colmap_name_to_image(colmap_object):
    name2content = {}
    for id, value in colmap_object.items():
        if value.name in name2content:
            logger.warning(f"File name {value.name} duplicate!")
        name2content[value.name] = value
    return name2content

def import_data_from_colmap_prior(img_lists, prior_colmap_model_path, single_camera=False):
    """ Import intrinsics and camera pose info """
    points3D_out = {}
    images_out = {}
    cameras_out = {}

    def compare(img_name):
        key = img_name.split('/')[-1]
        return key.split('.')[0]
    img_lists.sort(key=compare)

    key, img_id, camera_id = 1, 1, 1
    xys_ = np.zeros((0, 2), float) 
    point3D_ids_ = np.full(0, -1, int) # will be filled after triangulation 

    # Load prior colmap model:
    try:
        prior_cameras, prior_images, prior_points3D = read_model(prior_colmap_model_path, ext='.bin')
    except:
        prior_cameras, prior_images, prior_points3D = read_model(prior_colmap_model_path, ext='.txt')

    colmap_prior_name2images = convert_colmap_name_to_image(prior_images)

    if single_camera and len(prior_cameras) != 1:
        logger.warning(f"Prior model has {len(prior_cameras)}, switch single camera off automatically.")
        single_camera = False
    
    # import data
    for img_path in img_lists:
        
        # Read pose:
        img_name = img_path.split('/')[-1]
        # base_dir = osp.dirname(img_path).rstrip('color') # root dir of this sequence
        assert img_name in colmap_prior_name2images, f"{img_name} not exists in colmap prior model"

        qvec = colmap_prior_name2images[img_name].qvec
        tvec = colmap_prior_name2images[img_name].tvec

        # Read intrinsic
        colmap_camera = prior_cameras[colmap_prior_name2images[img_name].camera_id]
        camera_model = colmap_camera.model
        if camera_model == "PINHOLE":
            params = colmap_camera.params
        else:
            raise NotImplementedError
            
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        
        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=img_path,
            xys=xys_,
            point3D_ids=point3D_ids_
        )
        
        camera = Camera(
            id=camera_id,
            model='PINHOLE',
            width=w,
            height=h,
            params=params
        )
        
        images_out[key] = image
        cameras_out[camera_id] = camera

        key += 1
        img_id += 1

        if not single_camera:
            camera_id += 1
    
    return cameras_out, images_out, points3D_out

def import_data_from_poses_path(img_lists, poses_dir, prior_intrin_path, is_c2w=False, single_camera=True):
    """ Import intrinsics and camera pose info """
    points3D_out = {}
    images_out = {}
    cameras_out = {}

    assert osp.exists(prior_intrin_path)
    def compare(img_name):
        key = img_name.split('/')[-1]
        return key.split('.')[0]
    img_lists.sort(key=compare)

    key, img_id, camera_id = 1, 1, 1
    xys_ = np.zeros((0, 2), float) 
    point3D_ids_ = np.full(0, -1, int) # will be filled after triangulation 

    # import data
    # suppose the image_path can be formatted as  "/path/.../color/***.png"
    img_type = img_lists[0].split('/')[-2]
    for img_path in img_lists:
        
        img_name = osp.basename(img_path)
        img_base_name = osp.splitext(img_name)[0]
        # base_dir = osp.dirname(img_path).rstrip('color') # root dir of this sequence
        
        _, tvec, qvec = get_pose_from_txt(img_base_name, poses_dir, is_c2w=is_c2w)

        pose_base_dir = osp.dirname(poses_dir)
        if single_camera:
            assert osp.isfile(prior_intrin_path), f"single_camera is switched, however given a intrin directory"
            intrin_path = prior_intrin_path
        else:
            assert osp.isdir(prior_intrin_path), f"Provided intrinsics path is not a directory! You need to switch single_camera for providing only one intrinsic file "
            intrin_path = osp.join(prior_intrin_path, img_base_name+'.txt')

        with open(intrin_path, 'r') as f:
            lines = f.readlines()
            if lines[0][0] == '#':
                # colmap camera format
                data = lines[1].strip('\n').split()
                model, width, height, *params = data
                params = np.array(params, float)
                camera = Camera(
                    id=camera_id,
                    model=model,
                    width=int(width),
                    height=int(height),
                    params=params
                )
            else:
                K = np.loadtxt(intrin_path)
                # logger.info(f"Load intrin from: {intrin_path}")
                fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]
                    
                img = PIL.Image.open(img_path)  # does actually not load data
                w, h = img.size

                camera = Camera(
                    id=camera_id,
                    model='PINHOLE',
                    width=w,
                    height=h,
                    params=np.array([fx, fy, cx, cy])
                )
        
        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=img_name,
            xys=xys_,
            point3D_ids=point3D_ids_
        )
        
        
        images_out[key] = image
        cameras_out[camera_id] = camera

        key += 1
        img_id += 1
        if not single_camera:
            camera_id += 1
    
    return cameras_out, images_out, points3D_out


def generate_model(img_lists, empty_dir, prior_colmap_model_path=None, prior_pose_path=None, prior_intrin_path=None, pose_is_c2w=False, single_camera=True):
    """ Write intrinsics and camera poses into COLMAP format model"""
    logging.info('Generate empty model...')
    if prior_colmap_model_path is not None:
        logger.info(f"Load initial poses from colmap prior!")
        model = import_data_from_colmap_prior(img_lists, prior_colmap_model_path, single_camera)
    elif prior_pose_path is not None and prior_intrin_path is not None:
        logger.info(f"Load initial poses from pose path!")
        model = import_data_from_poses_path(img_lists, prior_pose_path, prior_intrin_path, is_c2w=pose_is_c2w, single_camera=single_camera)
    else:
        logger.error(f"Please provide colmap prior path or pose prior path for triangulation")
        raise NotImplementedError

    logging.info(f'Writing the COLMAP model to {empty_dir}')
    Path(empty_dir).mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(empty_dir), ext='.bin')
    write_model(*model, path=str(empty_dir), ext='.txt') # For easy visual
    logging.info('Finishing writing model.')
    