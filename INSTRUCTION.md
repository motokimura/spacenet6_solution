# spacenet6_solution
motokimura's solution to SpaceNet6 challenge

## Instructions for Final Scoring

This document provides instructions for the final testing/scoring phase.

### Download SpaceNet6 data

```
DATA_DIR=${HOME}/data  # path to download SpaceNet6 dataset
mkdir -p ${DATA_DIR}

# download and extract train data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz ${DATA_DIR}
tar -xvf SN6_buildings_AOI_11_Rotterdam_train.tar.gz

# download and extract test data
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz ${DATA_DIR}
tar -xvf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
```

### Prepare temporal directory

```
WDATA_DIR=${HOME}/wdata  # path to directory for temporal files (model weights, predictions, etc.)
mkdir -p ${WDATA_DIR}
```

### Build image

```
cd ${CODE_DIR}  # `code` directory containing `Dockerfile`, `train.sh`, `test.sh`, and etc. 
nvidia-docker build -t motokimura .
```

### Prepare container

```
# launch container
nvidia-docker run --ipc=host -v ${DATA_DIR}:/data:ro -v ${WDATA_DIR}:/wdata -it motokimura
```

It's necessary to add `--ipc=host` option when run docker (as written in [flags.txt](flags.txt)).
Otherwise multi-threaded PyTorch dataloader will crash.

### Train

```
# start training!
(in container) ./train.sh /data/train/AOI_11_Rotterdam

# if you need logs:
(in container) ./train.sh /data/train/AOI_11_Rotterdam 2>&1 | tee /wdata/train.log
```

Note that this is a sample call of `train.sh`. 
i.e., you need to specify the correct path to training data folder.

### Test

```
# start testing!
(in container) ./test.sh /data/test_public/AOI_11_Rotterdam /wdata/solution.csv

# if you need logs:
(in container) ./test.sh /data/test_public/AOI_11_Rotterdam /wdata/solution.csv 2>&1 | tee /wdata/test.log
```

Note that this is a sample call of `test.sh`. 
i.e., you need to specify the correct paths to testing image folder and output csv file.