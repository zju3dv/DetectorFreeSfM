import numpy as np
from kornia.utils.grid import create_meshgrid
import torch
import numpy as np
import torch.nn.functional as F
import math
from einops.einops import rearrange
from loguru import logger

def buildDataPair(data0, data1):
    # data0: dict, data1: dict
    data = {}
    for i, data_part in enumerate([data0, data1]):
        for key, value in data_part.items():
            data[key + str(i)] = value
    assert len(data) % 2 == 0, "build data pair error! please check built data pair!"
    data["pair_names"] = (data["img_path0"], data["img_path1"])
    return data