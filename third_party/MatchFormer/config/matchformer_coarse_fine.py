import sys
sys.path.append("third_party/MatchFormer/config")
from defaultmf import _CN as cfg

cfg.MATCHFORMER.BACKBONE_TYPE = 'largela'
cfg.MATCHFORMER.SCENS = 'outdoor'
cfg.MATCHFORMER.RESOLUTION = (8,2)
cfg.MATCHFORMER.COARSE.D_MODEL = 256
cfg.MATCHFORMER.COARSE.D_FFN = 256

cfg.MATCHFORMER.MATCH_COARSE.THR = 0.4 # 0.4 for eth3d tri
cfg.MATCHFORMER.FINE.ENABLE = True