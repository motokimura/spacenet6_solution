from yacs.config import CfgNode as CN


_C = CN()

# Input
_C.INPUT = CN()
_C.INPUT.TRAIN_DATA_LIST = '/data/spacenet6/split/train_0.json'
_C.INPUT.VAL_DATA_LIST = '/data/spacenet6/split/val_0.json'
_C.INPUT.IMAGE_DIR = '/data/spacenet6/spacenet6/train'
_C.INPUT.IMAGE_TYPE = 'SAR-Intensity'
_C.INPUT.BUILDING_DIR = '/data/spacenet6/footprint_boundary_mask/v_01/labels'
_C.INPUT.CLASSES = ['building_footprint', 'building_boundary']
_C.INPUT.MEAN_STD_DIR = '/data/spacenet6/image_mean_std/'
_C.INPUT.SAR_ORIENTATION = '/data/spacenet6/spacenet6/train/SummaryData/SAR_orientations.txt'

# Transforms
_C.TRANSFORM = CN()
_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE = (320, 320)
_C.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG = (-0, 0)
_C.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB = 0.0
_C.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB = 0.0
_C.TRANSFORM.TRAIN_SPECKLE_NOISE_STD = 0.0
_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD = 0.0
_C.TRANSFORM.TEST_SIZE = (928, 928)
_C.TRANSFORM.ALIGN_SAR_ORIENTATION = True
_C.TRANSFORM.TARGET_SAR_ORIENTATION = 0  # 0: north, 1: south

# Data loader
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 24
_C.DATALOADER.VAL_BATCH_SIZE = 16
_C.DATALOADER.TRAIN_NUM_WORKERS = 8
_C.DATALOADER.VAL_NUM_WORKERS = 8
_C.DATALOADER.TRAIN_SHUFFLE = True

# Model
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'unet'
_C.MODEL.BACKBONE = 'efficientnet-b5'
_C.MODEL.ENCODER_PRETRAINED_FROM = 'imagenet'
_C.MODEL.ACTIVATION = 'sigmoid'
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.IN_CHANNELS = 4
_C.MODEL.WEIGHT = ''

# Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 300
_C.SOLVER.OPTIMIZER = 'adam'  # ['adam', 'adamw']
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.LR = 1e-4
_C.SOLVER.LR_SCHEDULER = 'multistep'  # ['multistep', 'annealing']
_C.SOLVER.LR_MULTISTEP_MILESTONES = [270,]
_C.SOLVER.LR_MULTISTEP_GAMMA = 0.1
_C.SOLVER.LR_ANNEALING_T_MAX = 300
_C.SOLVER.LR_ANNEALING_ETA_MIN = 0.0
_C.SOLVER.LOSSES = ['dice', 'bce']  # ['dice', 'bce', 'focal']
_C.SOLVER.LOSS_WEIGHTS = [1.0, 1.0]
_C.SOLVER.FOCAL_LOSS_GAMMA = 2.0

# Eval
_C.EVAL = CN()
_C.EVAL.METRICS = ['iou',]  # ['iou']
_C.EVAL.MAIN_METRIC = 'iou/all'

# Misc
_C.LOG_ROOT = '/work/logs'
_C.LOG_DIR = 'exp_9999'


def get_default_config():
    return _C.clone()
