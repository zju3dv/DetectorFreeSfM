import pickle
import h5py
import os
import numpy as np
import torch
from tqdm import tqdm

def dict_to_cuda(data_dict):
    data_dict_cuda = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict_cuda[k] = v.cuda()
        elif isinstance(v, dict):
            data_dict_cuda[k] = dict_to_cuda(v)
        elif isinstance(v, list):
            data_dict_cuda[k] = list_to_cuda(v)
        else:
            data_dict_cuda[k] = v
    return data_dict_cuda

def list_to_cuda(data_list):
    data_list_cuda = []
    for obj in data_list:
        if isinstance(obj, torch.Tensor):
            data_list_cuda.append(obj.cuda())
        elif isinstance(obj, dict):
            data_list_cuda.append(dict_to_cuda(obj))
        elif isinstance(obj, list):
            data_list_cuda.append(list_to_cuda(obj))
        else:
            data_list_cuda.append(obj)
    return data_list_cuda

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_h5(file_path, transform_slash=True, parallel=False):
    """load the whole h5 file into memory (not memmaped)
    TODO: Loading data in parallel
    """
    with h5py.File(file_path, 'r') as f:
        # if parallel:
        #     Parallel()
        data = {k if not transform_slash else k.replace('+', '/'): v.__array__() \
                    for k, v in f.items()}
    return data

def save_h5(dict_to_save, filename, transform_slash=True, verbose=False, as_half=False):
    """Saves dictionary to hdf5 file"""
    with h5py.File(filename, 'w') as f:
        for key in tqdm(dict_to_save, disable=not verbose):  # h5py doesn't allow '/' in object name (will leads to sub-group)
            if as_half:
                dt = dict_to_save[key].dtype
                if (dt == np.float32) and (dt != np.float16):
                    data = dict_to_save[key].astype(np.float16)
                else:
                    data = dict_to_save[key]
                f.create_dataset(key.replace('/', '+') if transform_slash else key,
                                data=data)
            else:
                f.create_dataset(key.replace('/', '+') if transform_slash else key,
                                data=dict_to_save[key])


def load_calib(calib_fullpath_list, subset_index=None):
    """Load all IMC calibration files and create a dictionary."""

    calib = {}
    if subset_index is None:
        for _calib_file in calib_fullpath_list:
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace(
                "calibration_", ""
            )
            calib[img_name] = load_h5(_calib_file)
    else:
        for idx in subset_index:
            _calib_file = calib_fullpath_list[idx]
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace(
                "calibration_", ""
            )
            calib[img_name] = load_h5(_calib_file)
    return calib