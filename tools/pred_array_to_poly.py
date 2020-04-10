#!/usr/bin/env python3
# this script converts predicted arrays to building polygons
# in spacenet6 solution csv format.

import numpy as np
import os.path
import pandas as pd
import solaris as sol

from glob import glob
from tqdm import tqdm

from _init_path import init_path
init_path()

from spacenet6_model.configs import load_config
from spacenet6_model.utils import (
    compute_building_score, ensemble_subdir,
    gen_building_polys_using_contours, gen_building_polys_using_watershed,
    poly_filename
)


if __name__ == '__main__':
    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)

    out_dir = os.path.join(config.POLY_CSV_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=False)

    array_paths = glob(
        os.path.join(
            config.ENSEMBLED_PREDICTION_ROOT,
            subdir,
            '*.npy'
        )
    )
    array_paths.sort()

    firstfile = True

    for array_path in tqdm(array_paths):
        pred_array = np.load(array_path)

        footprint_channel = config.INPUT.CLASSES.index('building_footprint')
        boundary_channel = config.INPUT.CLASSES.index('building_boundary')

        footprint_score = pred_array[footprint_channel]
        boundary_score = pred_array[boundary_channel]
        building_score = compute_building_score(
            footprint_score,
            boundary_score,
            config.BOUNDARY_SUBSTRACT_COEFF
        )

        h, w = building_score.shape
        assert h == 900 and w == 900

        if config.METHOD_TO_MAKE_POLYGONS == 'contours':
            polys = gen_building_polys_using_contours(
                building_score,
                config.BUILDING_MIM_AREA_PIXEL,
                config.BUILDING_SCORE_THRESH
            )
        elif config.METHOD_TO_MAKE_POLYGONS == 'watershed':
            polys = gen_building_polys_using_watershed(
                building_score,
                config.WATERSHED_SEED_MIN_AREA_PIXEL,
                config.WATERSHED_MIN_AREA_PIXEL,
                config.WATERSHED_SEED_THRESH,
                config.WATERSHED_MAIN_THRESH
            )
        else:
            raise ValueError()

        if len(polys) == 0:
            polys = ["POLYGON EMPTY",]

        # add to the cumulative inference to dataframe
        filename = os.path.basename(array_path)
        tilename = '_'.join(os.path.splitext(filename)[0].split('_')[-4:])

        df = pd.DataFrame(
            {
                'ImageId': tilename,
                #'BuildingId': 0,
                'PolygonWKT_Pix': polys,
                'Confidence': 1  # TODO: compute confidence appropriately
            }
        )
        #df['BuildingId'] = range(len(df))
        if firstfile:
            building_poly_df = df
            firstfile = False
        else:
            building_poly_df = building_poly_df.append(df)

    building_poly_df.to_csv(
        os.path.join(out_dir, poly_filename()),
        index=False
    )
