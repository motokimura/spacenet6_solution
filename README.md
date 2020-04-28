# spacenet6_solution
a solution to spacenet6 challenge

## Usage

### Download SpaceNet6 data

```
# prepare data directory
DATA_DIR=${HOME}/data/spacenet6/spacenet6
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# download and extract train data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_train.tar.gz

# download and extract test data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
```

### Prepare training environment

```
# git clone source
PROJ_DIR=${HOME}/spacenet6_solution
git clone git@github.com:motokimura/spacenet6_solution.git ${PROJ_DIR}

# build docker image
cd ${PROJ_DIR}
./docker/build.sh

# launch docker container
ENV=desktop  # or "mac"
./docker/run.sh ${ENV}
```

### Preprocess dataset

All commands below have to be executed inside the container.

```
./tools/compute_mean_std.py --image_subdir SAR-Intensity

./tools/compute_mean_std.py --image_subdir PS-RGBNIR

./tools/geojson_to_mask.py

./tools/split_dataset.py

./tools/separate_val_images.py

./tools/separate_val_labels.py

# optionally you can create AMI here
```

### Train segmentation models

All commands below have to be executed inside the container.

```
EXP_ID=9999  # new experiment id
./tools/train_spacenet6_model.py [--config CONFIG_FILE] EXP_ID ${EXP_ID}
```

### Test segmentation models

All commands below have to be executed inside the container.

```
EXP_ID=9999  # previous experiment id from which config and weight are loaded
./tools/test_spacenet6_model.py [--config CONFIG_FILE] --exp_id ${EXP_ID}
```

### Ensemble segmentation models

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/ensemble_models.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

### Convert mask to polygon

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/pred_array_to_poly.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

### Test segmentation models (val images)

All commands below have to be executed inside the container.

```
EXP_ID=9999  # previous experiment id from which config and weight are loaded
./tools/test_spacenet6_model.py --config configs/test_to_val_images.yml --exp_id ${EXP_ID}
```

### Ensemble segmentation models (val images)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
SPLIT_ID=0  # split id of the models above
TEST_IMAGE_DIR=/data/spacenet6/val_images/val_${SPLIT_ID}/SAR-Intensity
./tools/ensemble_models.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS} INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR}
```

### Convert mask to polygon (val images)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/pred_array_to_poly.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

### Train LGBM models

All commands below have to be executed inside the container.

```
./tools/train_lgbm.py --truth_csvs ${TRUTH_CSVS} --solution_csvs ${SOLUTION_CSVS} --imageids ${IMAGEIDS} --pred_dirs ${PRED_DIRS} --out_dir ${OUT_DIR}
```

### Test LGBM models

All commands below have to be executed inside the container.

```
./tools/test_lgbm.py --solution_csv ${SOLUTION_CSV} --imageid ${IMAGEID} --pred_dir ${PRED_DIR} --models ${MODELS} --out ${OUT} [--iou_thresh ${IOU_THRESH}]
```


## Deployment

### Download SpaceNet6 data

```
DATA_DIR=${HOME}  # path to download SpaceNet6 dataset
cd ${DATA_DIR}

# download and extract train data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_train.tar.gz

# download and extract test data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
```

### Build image

```
cd ${CODE_DIR}  # `code` directory containing `Dockerfile`, `train.sh`, `test.sh`, and etc. 
docker build -t spacenet6 .
```

### Prepare container

```
# launch container
docker run --runtime nvidia -d -it --ipc=host --name spacenet6 spacenet6 /bin/bash

# copy SpaceNet6 data into the container (this may take twenty to thirty minutes)
docker cp ${DATA_DIR}/train spacenet6:/work/ && docker cp ${DATA_DIR}/test_public spacenet6:/work/
```

### Train

```
# enter the container
docker exec -it spacenet6 /bin/bash

# start training!
(in container) ./train.sh /work/train/AOI_11_Rotterdam

# if you need logs:
(in container) ./train.sh /work/train/AOI_11_Rotterdam 2>&1 | tee /work/train.log
```

### Test

```
# enter the container
docker exec -it spacenet6 /bin/bash

# start testing!
(in container) ./test.sh /work/test_public/AOI_11_Rotterdam /work/solution.csv

# if you need logs:
(in container) ./test.sh /work/test_public/AOI_11_Rotterdam /work/solution.csv 2>&1 | tee /work/test.log
```

After running commands above, you will find `/work/solution.csv` in the container.

You can specify the output path by the second argument of `test.sh`.
