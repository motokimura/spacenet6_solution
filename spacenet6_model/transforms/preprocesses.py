
import albumentations as albu
import functools
import numpy as np
import os.path


def get_spacenet6_preprocess(config):
    """
    """
    mean_path = os.path.join(
        config.INPUT.MEAN_STD_DIR,
        config.INPUT.IMAGE_TYPE,
        'mean.npy'
    )
    mean = np.load(mean_path)
    mean = mean[np.newaxis, np.newaxis, :]

    std_path = os.path.join(
        config.INPUT.MEAN_STD_DIR,
        config.INPUT.IMAGE_TYPE,
        'std.npy'
    )
    std = np.load(std_path)
    std = std[np.newaxis, np.newaxis, :]

    preprocess = [
        albu.Lambda(
            image=functools.partial(
                _normalize_image,
                mean=mean,
                std=std
            )
        ),
        albu.Lambda(
            image=functools.partial(_to_tensor),
            mask=functools.partial(_to_tensor)
        ),
    ]
    return albu.Compose(preprocess)


def _normalize_image(image, mean, std, **kwargs):
    """
    """
    normalized = image.astype('float32')
    normalized = (image - mean) / std
    return normalized


def _to_tensor(x, **kwargs):
    """
    """
    return x.transpose(2, 0, 1).astype('float32')
