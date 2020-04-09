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
./tools/train.py [--config CONFIG_FILE] EXP_ID ${EXP_ID}
```

### Test segmentation models

All commands below have to be executed inside the container.

```
EXP_ID=9999  # previous experiment id from which config and weight are loaded
./tools/test.py [--config CONFIG_FILE] --exp_id ${EXP_ID}
```

### Ensemble segmentation models

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999, 9998, 9997, 9996, 9995]'  # previous experiments used for ensemble
./tools/ensemble_models.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

### Convert mask to polygon

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999, 9998, 9997, 9996, 9995]'  # previous experiments used for ensemble
./tools/pred_array_to_poly.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Deployment

### Download SpaceNet6 data

```
# download and extract train data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_train.tar.gz

# download and extract test data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
```

### Setup container

```
git clone git@github.com:motokimura/spacenet6_solution.git
cd spacenet6_solution
docker build -t spacenet6 .

cd ..
rm -rf spacenet6_solution

docker run --runtime nvidia -d -it --name spacenet6 spacenet6 /bin/bash
```

### Copy data into container

```
echo 'Copying train dataset...'
docker cp train spacenet6:/work/

echo 'Copying test dataset...'
docker cp test_public spacenet6:/work/

# optionally you can create AMI here
```

### Train

```
docker exec -it spacenet6 /bin/bash

./train.sh train/AOI_11_Rotterdam
```

### Test

```
docker exec -it spacenet6 /bin/bash

./test.sh test_public/AOI_11_Rotterdam solution.csv
```
