# BDLoss
The official implementation of the batch-wise dice loss.

## Environment
Build the conda environment for this project.
```
conda env create -f env.yml
conda activate BDLoss
```

## Dataset
We use two publicly accessible datasets for model training and evaluation.
### VIPCUP 2018 
website: http://i-sip.encs.concordia.ca/2018VIP-Cup/index.html

You can download the dataset and process by yourself. Or download the processed data directly with the following command:
```
wget -O ./VIPCUPData.zip "https://gntuedutw-my.sharepoint.com/:u:/g/personal/r07922058_g_ntu_edu_tw/Ea3lZ6yFcm9OkO2kzD_l2D4BCO0UhmAZjsMR-HOob3hExA?download=1"
```

### Kits 2019
website: https://kits19.grand-challenge.org/
You need to download the data from the official github repo and run `kits_preprocessing.py`
```
python -m src.kits_preprocessing DATA_PATH SAVED_DATA_PATH ./src/kits_data_split.csv
```

## Training
You have to modify the config files under the `configs/train` folder first.
```
python -m src.main configs/train/dense_highres3dnet_bdloss_vipcup_seg_config.yaml
python -m src.main configs/train/dense_highres3dnet_bdloss_kits_seg_config.yaml
```

## Testing
You have to modify the config files under the `configs/test` folder first.
```
python -m src.inference configs/test/dense_highres3dnet_bdloss_vipcup_seg_config.yaml
python -m src.inference configs/test/dense_highres3dnet_bdloss_kits_seg_config.yaml
```
