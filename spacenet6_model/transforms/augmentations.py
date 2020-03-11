import albumentations as albu
import functools
import numpy as np


def get_spacenet6_augmentation(config, is_train):
    """
    """
    if is_train:
        augmentation = [
            # random flip
            albu.HorizontalFlip(
                p=config.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB
            ),
            albu.VerticalFlip(
                p=config.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB
            ),
            # random rotate
            albu.ShiftScaleRotate(
                scale_limit=0.0,
                rotate_limit=config.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG,
                shift_limit=0.0,
                p=1,
                border_mode=0),
            # random crop
            albu.RandomCrop(
                width=config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[0],
                height=config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[1],
                always_apply=True
            ),
            # speckle noise
            albu.Lambda(
                image=functools.partial(
                    _random_speckle_noise,
                    speckle_std=config.TRANSFORM.TRAIN_SPECKLE_NOISE_STD
                )
            ),
            # random brightness
            albu.Lambda(
                image=functools.partial(
                    _random_brightness,
                    brightness_std=config.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD
                )
            ),
        ]
    else:
        augmentation = [
            albu.PadIfNeeded(
                min_width=config.TRANSFORM.TEST_SIZE[0],
                min_height=config.TRANSFORM.TEST_SIZE[1],
                always_apply=True,
                border_mode=0
            )
        ]
    return albu.Compose(augmentation)


def _random_speckle_noise(image, speckle_std, **kwargs):
    """
    """
    if speckle_std <= 0:
        return image

    im_shape = image.shape
    gauss = np.random.normal(0, speckle_std, im_shape)
    gauss = gauss.reshape(im_shape)
    speckle_noise = gauss * image
    noised = image + speckle_noise

    return noised


def _random_brightness(image, brightness_std, **kwargs):
    """
    """
    if brightness_std <= 0:
        return image

    gauss = np.random.normal(0, brightness_std)
    brightness_noise = gauss * image
    noised = image + brightness_noise

    return noised
