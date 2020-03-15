import json
import numpy as np
import os.path

from skimage import io
from torch.utils.data import Dataset

from .utils import read_orientation_file, lookup_orientation


class SpaceNet6Dataset(Dataset):
    """
    """
    CLASSES = [
        'background',          # mask_val = 0
        'building_footprint',  # mask_val = 1
        'building_boundary'    # mask_val = 2
    ]

    def __init__(
        self,
        config,
        data_list_path,
        augmentation=None,
        preprocessing=None
    ):
        # generate full path to image/label files
        data_list = self._get_data_list(data_list_path)

        image_root = config.INPUT.IMAGE_DIR
        image_type = config.INPUT.IMAGE_TYPE
        image_dir = os.path.join(image_root, image_type)
        mask_dir = config.INPUT.BUILDING_DIR

        self.image_paths, self.mask_paths = [], []
        for data in data_list:
            self.image_paths.append(os.path.join(image_dir, data[image_type]))
            self.mask_paths.append(os.path.join(mask_dir, data['Mask']))

        # prepare sar orientation look up table to align orientation of all SAR images
        assert config.TRANSFORM.TARGET_SAR_ORIENTATION in [0, 1]  # north (0) or south (1)
        if image_type == 'SAR-Intensity' and config.TRANSFORM.ALIGN_SAR_ORIENTATION:
            self.orientation_df = read_orientation_file(
                os.path.join(image_root, 'SummaryData/SAR_orientations.txt')
            )
            self.target_orientation = config.TRANSFORM.TARGET_SAR_ORIENTATION
        else:
            self.orientation_df = None
            self.target_orientation = None

        # convert str names to class values on masks
        classes = config.INPUT.CLASSES
        if not classes:
            # if classes is empty, use all classes
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        """
        image = io.imread(self.image_paths[i])
        mask = io.imread(self.mask_paths[i])

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.orientation_df is not None:
            orientation = lookup_orientation(
                self.image_paths[i],
                self.orientation_df
            )
            if orientation != self.target_orientation:
                # align orientation
                image = np.fliplr(np.flipud(image))
                mask = np.fliplr(np.flipud(mask))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """
        """
        return len(self.image_paths)

    def _get_data_list(self, data_list_path):
        """
        """
        with open(data_list_path) as f:
            data_list = json.load(f)
        return data_list
