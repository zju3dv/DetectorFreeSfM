import os.path as osp
import numpy as np
from pathlib import Path
import scipy.spatial.distance as distance
from src.colmap.read_write_model import qvec2rotmat
from src.utils.colmap.read_write_model import read_images_binary

def write_fixed_images(image_list, output_path):
    image_name_0 = osp.basename(image_list[0])
    image_name_1 = osp.basename(image_list[(len(image_list) // 2)])
    image_names = [image_name_0, image_name_1]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for image_name in image_names:
            f.write(image_name)
            f.write('\n')

def get_pairwise_distance(poses_list):
    poses = np.stack(poses_list)
    Rs = poses[:, :3, :3]  # N*3*3
    ts = poses[:, :3, [3]] # N*3*1

    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts)[:, :, 0] # N*3

    dist = distance.squareform(distance.pdist(ts)) # N, N

    return dist

def fix_farest_images(reconstructed_model_dir, output_path):
    # Load images and poses
    colmap_images = read_images_binary(osp.join(reconstructed_model_dir, 'images.bin'))
    pose_list = []
    id_list = []
    for id, image in colmap_images.items():
        R = qvec2rotmat(image.qvec) # 3*3
        t = image.tvec # 3
        pose = np.eye(4) # 4*4
        pose[:3, :3] = R
        pose[:3, 3] = t

        pose_list.append(pose)
        id_list.append(id)
    
    dist = get_pairwise_distance(pose_list) # N*N

    # Get farest image pair
    index = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    colmap_id0, colmap_id1 = id_list[index[0]], id_list[index[1]]
    image_names = [colmap_images[colmap_id0].name, colmap_images[colmap_id1].name]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for image_name in image_names:
            f.write(image_name)
            f.write('\n')

def fix_all_images(reconstructed_model_dir, output_path):
    # Load images and poses:
    colmap_images = read_images_binary(osp.join(reconstructed_model_dir, 'images.bin'))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for colmap_image in colmap_images.values():
            f.write(colmap_image.name)
            f.write('\n')