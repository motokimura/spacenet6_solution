from yacs.config import CfgNode as CN


_C = CN()

# Input
_C.INPUT = CN()
_C.INPUT.TRAIN_VAL_SPLIT_DIR = '/data/spacenet6/split'
_C.INPUT.TRAIN_VAL_SPLIT_ID = 0
_C.INPUT.IMAGE_DIR = '/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/'
_C.INPUT.IMAGE_TYPE = 'SAR-Intensity'
_C.INPUT.BUILDING_DIR = '/data/spacenet6/footprint_boundary_mask/v_01/labels'
_C.INPUT.SAR_ORIENTATION = '/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt'
_C.INPUT.CLASSES = ['building_footprint', 'building_boundary']
_C.INPUT.MEAN_STD_DIR = '/data/spacenet6/image_mean_std/'
_C.INPUT.TEST_IMAGE_DIR = '/data/spacenet6/spacenet6/test_public/AOI_11_Rotterdam/SAR-Intensity'

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
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.TRAIN_NUM_WORKERS = 8
_C.DATALOADER.VAL_NUM_WORKERS = 8
_C.DATALOADER.TEST_NUM_WORKERS = 8
_C.DATALOADER.TRAIN_SHUFFLE = True


# Model
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'unet'  # ['unet', 'fpn', 'pan', 'pspnet', 'deeplabv3', 'linknet']
_C.MODEL.BACKBONE = 'efficientnet-b5'
_C.MODEL.ENCODER_PRETRAINED_FROM = 'imagenet'
_C.MODEL.ACTIVATION = 'sigmoid'
_C.MODEL.IN_CHANNELS = 4
_C.MODEL.UNET_ENABLE_DECODER_SCSE = False
_C.MODEL.FPN_DECODER_DROPOUT = 0.2
_C.MODEL.PSPNET_DROPOUT = 0.2
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.WEIGHT = 'none'

# Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 260
_C.SOLVER.OPTIMIZER = 'adam'  # ['adam', 'adamw']
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.LR = 1e-4
_C.SOLVER.LR_SCHEDULER = 'multistep'  # ['multistep', 'annealing']
_C.SOLVER.LR_MULTISTEP_MILESTONES = [230,]
_C.SOLVER.LR_MULTISTEP_GAMMA = 0.1
_C.SOLVER.LR_ANNEALING_T_MAX = 260
_C.SOLVER.LR_ANNEALING_ETA_MIN = 0.0
_C.SOLVER.LOSSES = ['dice', 'bce']  # ['dice', 'bce', 'focal']
_C.SOLVER.LOSS_WEIGHTS = [1.0, 1.0]
_C.SOLVER.FOCAL_LOSS_GAMMA = 2.0

# Eval
_C.EVAL = CN()
_C.EVAL.METRICS = ['iou',]  # ['iou']
_C.EVAL.MAIN_METRIC = 'iou/building_footprint'

# Misc
_C.LOG_ROOT = '/logs'
_C.WEIGHT_ROOT = '/weights'
_C.CHECKPOINT_ROOT = '/checkpoints'
_C.PREDICTION_ROOT = '/predictions'
_C.ENSEMBLED_PREDICTION_ROOT = '/ensembled_predictions'
_C.POLY_CSV_ROOT = '/polygons'
_C.POLY_OUTPUT_PATH = 'none'
_C.SAVE_CHECKPOINTS = True
_C.DUMP_GIT_INFO = True
_C.TEST_TO_VAL = False
_C.BOUNDARY_SUBSTRACT_COEFF = 0.2
_C.METHOD_TO_MAKE_POLYGONS = 'watershed'  # ['contours', 'watershed']
_C.BUILDING_SCORE_THRESH = 0.5
_C.BUILDING_MIM_AREA_PIXEL = 0
_C.WATERSHED_MAIN_THRESH = 0.3
_C.WATERSHED_SEED_THRESH = 0.7
_C.WATERSHED_MIN_AREA_PIXEL = 80
_C.WATERSHED_SEED_MIN_AREA_PIXEL = 20
_C.EXP_ID = 9999  # 0~9999
_C.ENSEMBLE_EXP_IDS = []


def get_default_config():
    return _C.clone()
