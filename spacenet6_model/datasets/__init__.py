import torch.utils.data

from .spacenet6 import SpaceNet6Dataset
from ..transforms import get_augmentation, get_preprocess


def get_dataloader(
    config,
    is_train
):
    """
    """
    preprocessing = get_preprocess(config)
    augmentation = get_augmentation(config, is_train)

    if is_train:
        data_list_path = config.INPUT.TRAIN_DATA_LIST
        batch_size = config.DATALOADER.TRAIN_BATCH_SIZE
        num_workers = config.DATALOADER.TRAIN_NUM_WORKERS
        shuffle = config.DATALOADER.TRAIN_SHUFFLE
    else:
        data_list_path = config.INPUT.VAL_DATA_LIST
        batch_size = config.DATALOADER.VAL_BATCH_SIZE
        num_workers = config.DATALOADER.VAL_NUM_WORKERS
        shuffle = False

    dataset = SpaceNet6Dataset(
        config,
        data_list_path,
        augmentation=augmentation,
        preprocessing=preprocessing
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
