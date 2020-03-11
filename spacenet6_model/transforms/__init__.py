from .augmentations import get_spacenet6_augmentation
from .preprocesses import get_spacenet6_preprocess


def get_preprocess(config):
    """
    """
    return get_spacenet6_preprocess(config)


def get_augmentation(config, is_train):
    """
    """
    return get_spacenet6_augmentation(config, is_train)
