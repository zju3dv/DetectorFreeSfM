import sys
import os.path as osp
from pathlib import Path
# sys.path.append("third_party/MatchFormer/config")
sys.path.append(osp.join((str(Path(__file__)).split('third_party')[0]), "third_party/MatchFormer/config"))
from defaultmf import _CN as cfg

cfg.MATCHFORMER.BACKBONE_TYPE = 'largela'
cfg.MATCHFORMER.SCENS = 'outdoor'
cfg.MATCHFORMER.RESOLUTION = (8,2)
cfg.MATCHFORMER.COARSE.D_MODEL = 256
cfg.MATCHFORMER.COARSE.D_FFN = 256

cfg.MATCHFORMER.MATCH_COARSE.THR = 0.4 # For eth3d tri
# cfg.MATCHFORMER.MATCH_COARSE.THR = 0.2 # For eth3d recon
cfg.MATCHFORMER.FINE.ENABLE = False