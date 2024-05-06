import io
import math
import h5py
from os import path as osp
from loguru import logger

import cv2
import torch
import numpy as np
import PIL
import albumentations as A


def process_resize(w, h, resize, df=None, resize_no_larger_than=False):
    assert(len(resize) > 0 and len(resize) <= 2)
    if resize_no_larger_than and (max(h, w) <= max(resize)):
        w_new, h_new = w, h
    else:
        if len(resize) == 1 and resize[0] > -1:  # resize the larger side
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    pad_size_h, pad_size_w = pad_size, pad_size

    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
            mask[:inp.shape[0], :inp.shape[1]] = 1
    elif inp.ndim == 3:
        padded = np.zeros((pad_size_h, pad_size_w, inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
            mask[:inp.shape[0], :inp.shape[1]] = 1
    else:
        raise NotImplementedError()
    return padded, mask


def grayscale2tensor(image, mask=None):
    return torch.from_numpy(image/255.).float()[None]  # (1, h, w)

def rgb2tensor(image):
    return torch.from_numpy(image/255.).float().permute(2, 0, 1).contiguous()  # (3, h, w)

def mask2tensor(mask):
    return torch.from_numpy(mask).float()  # (h, w)


def depth2tensor(depth):
    # don't do normalization
    return torch.from_numpy(depth).float()  # (h, w)


def ndarray2grayscale(inp):
    """Transform ndarray (from tensor.cpu().numpy) to grayscale image"""
    return (inp[0] * 255).round().astype(np.int32)

def load_intrinsics_from_h5(intrinsics_path):
    assert osp.exists(intrinsics_path), f"intrinsic path {intrinsics_path} not exists"
    with h5py.File(intrinsics_path, 'r') as f:
        intrinsics = f['K'].__array__().astype(np.float32)
    return intrinsics


def read_rgb(path, resize=None, resize_no_larger_than=False, resize_float=False, df=None, client=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    resize = tuple(resize) if resize is not None else None
    if augmentor is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_COLOR)  # BGR image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentor(image)

    if image is None:
        logger.error(f"Problem exists when loading image: {path}")
    
    # import ipdb; ipdb.set_trace()
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize if resize is not None else (w, h), df, resize_no_larger_than=resize_no_larger_than)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]
    original_hw = torch.tensor([h,w]) #[2]

    image = resize_image(image, (w_new, h_new), interp="pil_LANCZOS").astype('float32')

    if pad_to is not None:
        if pad_to == -1:
            pad_to = max(w_new, h_new)

        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    ts_image = rgb2tensor(image)
    ret_val = [ts_image]

    if ret_scales:
        ret_val += [scales, original_hw]
    if ret_pad_mask:
        ts_mask = mask2tensor(mask) if pad_to else None
        ret_val.append(ts_mask if pad_to else None)
    return ret_val[0] if len(ret_val) == 1 else ret_val

def read_grayscale(path, resize=None, resize_no_larger_than=False, resize_float=False, df=None, client=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    resize = tuple(resize) if resize is not None else None
    if augmentor is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_COLOR)  # BGR image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentor(image)

    if image is None:
        logger.error(f"Problem exists when loading image: {path}")

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize if resize is not None else (w, h), df, resize_no_larger_than=resize_no_larger_than)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]
    original_hw = torch.tensor([h,w]) #[2]

    image = resize_image(image, (w_new, h_new), interp="pil_LANCZOS").astype('float32')

    if pad_to is not None:
        if pad_to == -1:
            pad_to = max(w_new, h_new)

        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    ts_image = grayscale2tensor(image)
    ret_val = [ts_image]

    if ret_scales:
        ret_val += [scales, original_hw]
    if ret_pad_mask:
        ts_mask = mask2tensor(mask) if pad_to else None
        ret_val.append(ts_mask if pad_to else None)
    return ret_val[0] if len(ret_val) == 1 else ret_val


def resize_image(image, size, interp):
    # NOTE: from hloc
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

def read_color(path, client):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
        else load_array_from_petrel(path, client, cv2.IMREAD_COLOR)
    return image


def read_grayscale_megadepth(path, client=None, ret_tensor=True):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if client is None \
        else load_array_from_petrel(path, client, cv2.IMREAD_GRAYSCALE)

    return grayscale2tensor(image) if ret_tensor else image


def read_depth(path, client=None):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if client is None \
        else load_array_from_petrel(path, client, cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    ts_depth = depth2tensor(depth)
    return ts_depth


def read_depth_megadepth(path, client=None, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['/depth']) if client is None \
        else load_array_from_petrel(path, client, None, use_h5py=True)  # (h, w)

    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to)

    return depth2tensor(depth)

def read_kpts_megadepth(path,client=None):
    if client is None:
        assert osp.exists(path), "file not exist!"
        with h5py.File(path,"r") as f:
            keypoints=f["/keypoints"].__array__().astype(np.float32)
            keypoints_img_size=f["/img_size"].__array__().astype(np.float32)
    else:
        keypoints,keypoints_img_size=load_kpts_from_petrel(path,client)
    
    return keypoints,keypoints_img_size

def load_kpts_from_petrel(path,client=None):
    byte_str = client.Get(path)
    try:
        f = io.BytesIO(byte_str)
        keypoints = np.array(h5py.File(f, 'r')['/keypoints']).astype(np.float32)
        keypoints_img_size = np.array(h5py.File(f,'r')['/img_size']).astype(np.float32)
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex
    assert keypoints is not None
    return keypoints,keypoints_img_size

def load_array_from_petrel(path, client, cv_type, max_retry=10, use_h5py=False, return_tensor=True):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def calc_max_size(img_path, max_area):
    try:
        h, w = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
    except AttributeError as err:
        logger.error(f'Image loading failed: {img_path}')
        raise AttributeError from err
        
    max_scale = min(math.sqrt(max_area / (h*w)), 1.)
    return max(h, w) * max_scale


def build_fake_input(size, dtype=torch.float32):
    fake = torch.randn(1, 1, size, size, dtype=dtype)
    scale = torch.tensor([[1., 1.]], dtype=dtype)
    return {'image0': fake, 'image1': fake, 'scale0': scale, 'scale1': scale}
