from yacs.config import CfgNode as CN
_CN = CN()

_CN.MATCHFORMER = CN()
_CN.MATCHFORMER.BACKBONE_TYPE = 'largela'# litela,largela,litesea,largesea
_CN.MATCHFORMER.SCENS = 'indoor' # indoor, outdoor
_CN.MATCHFORMER.RESOLUTION = (8,2)  #(8,2),(8,4)
_CN.MATCHFORMER.FINE_WINDOW_SIZE = 5
_CN.MATCHFORMER.FINE_CONCAT_COARSE_FEAT = True

_CN.MATCHFORMER.COARSE = CN()
_CN.MATCHFORMER.COARSE.D_MODEL = 256
_CN.MATCHFORMER.COARSE.D_FFN = 256

_CN.MATCHFORMER.MATCH_COARSE = CN()
_CN.MATCHFORMER.MATCH_COARSE.THR = 0.2
_CN.MATCHFORMER.MATCH_COARSE.BORDER_RM = 0
_CN.MATCHFORMER.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
_CN.MATCHFORMER.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCHFORMER.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCHFORMER.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCHFORMER.MATCH_COARSE.SKH_PREFILTER = False
_CN.MATCHFORMER.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2
_CN.MATCHFORMER.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200
_CN.MATCHFORMER.MATCH_COARSE.SPARSE_SPVS = True

_CN.MATCHFORMER.FINE = CN()
_CN.MATCHFORMER.FINE.D_MODEL = 128
_CN.MATCHFORMER.FINE.D_FFN = 128
_CN.MATCHFORMER.FINE.ENABLE = True

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# geometric metrics and pose solver
_CN.TRAINER = CN()
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
_CN.TRAINER.SEED = 66

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
