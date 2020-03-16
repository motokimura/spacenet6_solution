#!/usr/bin/env python3
import numpy as np
import os.path

from _init_path import init_path
init_path()

from spacenet6_model.configs import load_config
from spacenet6_model.datasets import get_test_dataloader
from spacenet6_model.models import get_model
from spacenet6_model.utils import crop_center


def main():
    """
    """
    config = load_config()
    print('successfully loaded config:')
    print(config)

    # prepare dataloader
    test_dataloader = get_test_dataloader(config)

    # prepare model to test
    model = get_model(config)
    model.eval()

    # test loop
    for batch in test_dataloader:
        images = batch['image']
        rotated_flags = batch['rotated']
        image_paths = batch['image_path']
        original_heights, original_widths, _ = batch['original_shape']

        predictions = model.module.predict(images)
        predictions = predictions.cpu().numpy()

        for i in range(len(predictions)):
            pred = predictions[i]
            path = image_paths[i]
            rotated = rotated_flags[i].item()
            orig_h = original_heights[i].item()
            orig_w = original_widths[i].item()

            assert orig_w == 900 and orig_h == 900

            c, h, w = pred.shape
            assert c == len(config.INPUT.CLASSES)
            assert h == config.TRANSFORM.TEST_SIZE[0]
            assert w == config.TRANSFORM.TEST_SIZE[1]

            # remove padded area
            pred = crop_center(pred, crop_wh=(orig_w, orig_h))

            if rotated:
                # re-rotate if input image is rotated
                pred = np.flipud(np.fliplr(pred))

            c, h, w = pred.shape
            assert c == len(config.INPUT.CLASSES)
            assert h == 900
            assert w == 900

            # dump to .npy file
            filename = os.path.basename(path)
            filename, _ = os.path.splitext(filename)
            filename = f'{filename}.npy'
            np.save(
                os.path.join(config.PREDICTION_ROOT, filename),
                pred
            )


if __name__ == '__main__':
    main()
