import cv2
import numpy as np
import os
import pandas as pd
import solaris as sol

from geomet import wkt
from skimage import io, measure
from tqdm import tqdm

from ..datasets.utils import lookup_orientation


# evaluate solution csv to get labels (iou_score of each polygon) to train LGBM
def evaluate_solution_csv(truth_csv, solution_csv):
    """
    """
    evaluator = sol.eval.base.Evaluator(truth_csv)
    evaluator.load_proposal(solution_csv, conf_field_list=['Confidence'], proposalCSV=True)
    evaluator.eval_iou_spacenet_csv(min_area=0)
    return evaluator.proposal_GDF


# compute features and label of each polygon
def generate_polygon_mask(image_df, image_size):
    """
    """
    def draw_mask_for_a_polygon(contours, image_size, val):
        """
        """
        mask = np.zeros(shape=image_size, dtype=np.uint8)
        for i, c in enumerate(contours):
            c = np.array(c).astype(np.int)
            color = val if i == 0 else 0
            mask = cv2.fillPoly(mask, [c], color=(color,))
        return mask

    mask = -np.ones(shape=image_size, dtype=np.int)  # background: -1
    for idx, p in enumerate(image_df['PolygonWKT_Pix']):
        contours = wkt.loads(p)['coordinates']
        assert len(contours) >= 1    
        mask_for_a_poly = draw_mask_for_a_polygon(
            contours,
            image_size=image_size,
            val=255
        )
        mask[mask_for_a_poly > 0] = idx
    return mask


def extract_polygons_from_mask(mask):
    """
    """
    polys = []
    
    N_polys = mask.max() + 1
    for idx in range(N_polys):
        mask_for_a_poly = (mask == idx).astype(np.uint8) * 255
        
        # get polygon contour from mask
        labels = measure.label(mask_for_a_poly, connectivity=2, background=0).astype('uint16')
        props = measure.regionprops(labels)
        assert len(props) >= 1
        prop = props[0]
        
        polys.append(prop)

    return polys


def compute_features(df, image_dir, rotation_df, image_size=(900, 900), is_train=True):  # XXX: image size is hard coded
    """
    """
    xs = []  # features
    
    for image_id in tqdm(df['ImageId'].unique()):
        # dataframe pocessing polygons for an image
        image_df = df[df['ImageId'] == image_id]
        
        # skip this image if empty
        if image_df['PolygonWKT_Pix'].iloc[0] == 'POLYGON EMPTY':
            continue
        
        # generate polygon mask image
        mask = generate_polygon_mask(image_df, image_size)

        prefix = 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity' if is_train \
            else 'SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity'

        # load SAR image
        image_path = os.path.join(
            image_dir,
            f'{prefix}_{image_id}.tif'
        )
        image = io.imread(image_path)
        
        # align orientation of image/mask to north (=0)
        rot = lookup_orientation(image_path, rotation_df)
        assert rot in [0, 1]
        if rot == 1:
            image = np.fliplr(np.flipud(image))
            mask = np.fliplr(np.flipud(mask))
        
        # compute raw features of each polygon
        polys = extract_polygons_from_mask(mask)
        areas = np.array([p.area for p in polys])
        centroids = np.array([p.centroid for p in polys])

        # extract features from each polygon
        N_polys = len(polys)
        for idx in range(N_polys):
            x = []  # feature

            prop = polys[idx]
            area = areas[idx]
            centroid = centroids[idx]

            # append features
            on_border = ((prop.bbox[0] <= 1) | (prop.bbox[1] <= 1) | (prop.bbox[2] >= image_size[0] - 1) | (prop.bbox[3] >= image_size[1] - 1))
            on_border *= 1  # bool to int
            x.append(on_border)
            x.append(area)
            x.append(prop.convex_area)
            x.append(prop.solidity)
            x.append(prop.eccentricity)
            x.append(prop.extent)
            x.append(prop.major_axis_length)
            x.append(prop.minor_axis_length)
            x.append(prop.euler_number)
            x.append(prop.equivalent_diameter)
            assert prop.major_axis_length > 0
            x.append(prop.minor_axis_length / prop.major_axis_length)
            x.append(prop.perimeter ** 2 / (4 * area * np.pi))
            
            # min_area_rect related feature
            mask_for_a_poly = (mask == idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_for_a_poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) >= 1
            cnt = contours[0]
            min_area_rect = cv2.minAreaRect(cnt)

            x.append(min(min_area_rect[1]))
            x.append(max(min_area_rect[1]))
            assert max(min_area_rect[1]) > 0
            x.append(min(min_area_rect[1]) / max(min_area_rect[1]))
            x.append(min_area_rect[2])
            x.append(1 * cv2.isContourConvex(cnt))
            
            # image feature
            mean = image[mask_for_a_poly > 0].mean(axis=0)  # [4,]
            std = image[mask_for_a_poly > 0].std(axis=0)  # [4,]
            
            x.extend(list(mean))
            x.extend(list(std))
            
            # relationships with other polygons
            idxs_other = np.ones(N_polys, dtype=bool)
            idxs_other[idx] = False
            centroids_other = centroids[idxs_other]
            areas_other = areas[idxs_other]
            
            d = np.sqrt(((centroids_other - centroid) ** 2.0).sum(axis=1))
            neighbers_200px = (d <= 200)
            neighbers_400px = (d <= 400)
            neighbers_600px = (d <= 600)
            neighbers_800px = (d <= 800)
            
            N_200px = neighbers_200px.sum()
            N_400px = neighbers_400px.sum()
            N_600px = neighbers_600px.sum()
            N_800px = neighbers_800px.sum()

            med_area_200px = np.median(areas_other[neighbers_200px]) if N_200px > 0 else -1.
            med_area_400px = np.median(areas_other[neighbers_400px]) if N_400px > 0 else -1.
            med_area_600px = np.median(areas_other[neighbers_600px]) if N_600px > 0 else -1.
            med_area_800px = np.median(areas_other[neighbers_800px]) if N_800px > 0 else -1.
            
            area_ratio_200px = area / med_area_200px if N_200px > 0 else -1.
            area_ratio_400px = area / med_area_400px if N_400px > 0 else -1.
            area_ratio_600px = area / med_area_600px if N_600px > 0 else -1.
            area_ratio_800px = area / med_area_800px if N_800px > 0 else -1.
            
            x.append(N_200px)
            x.append(N_400px)
            x.append(N_600px)
            x.append(N_800px)
            
            x.append(med_area_200px)
            x.append(med_area_400px)
            x.append(med_area_600px)
            x.append(med_area_800px)
            
            x.append(area_ratio_200px)
            x.append(area_ratio_400px)
            x.append(area_ratio_600px)
            x.append(area_ratio_800px)

            # length check
            assert len(x) == 37
            xs.append(x)
    
    return np.array(xs)


def get_labels(df):
    """
    """
    ys = []
    for image_id in tqdm(df['ImageId'].unique()):
        # dataframe pocessing polygons for an image
        image_df = df[df['ImageId'] == image_id]
        # skip this image if empty
        if image_df['PolygonWKT_Pix'].iloc[0] == 'POLYGON EMPTY':
            continue
        # extract iou score from each polygon
        for iou in image_df['iou_score']:
            ys.append(iou)
    return np.array(ys)


# prepare dataset to train LGBM
def get_dataset(features, labels, df_idxs):
    """
    """
    assert len(features) == len(labels)
    ret_features, ret_labels = [], []
    for idx in df_idxs:
        ret_features.append(features[idx])
        ret_labels.append(labels[idx])
    ret_features = np.concatenate(ret_features, axis=0)
    ret_labels = np.concatenate(ret_labels, axis=0)
    return ret_features, ret_labels


# utils for LGBM inference
def image_has_no_polygon(image_df):
    p = image_df['PolygonWKT_Pix']
    if len(p) > 1:
        return False
    return (p.values[0] == 'POLYGON EMPTY')


def get_lgmb_prediction(models, x):
    preds = []
    for m in models:
        pred = m.predict(x)
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0)  # ensemble
