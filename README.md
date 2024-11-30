# DAG-UNet

## Data

* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the same folder path as the code for training.
* Busi Data - The data is available at [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). Create a new folder named 'inputs', and keep the downloaded data to the folder. For data with more than one mask, we add the masks together.

## Training
```
cd into DAG-UNet
```

For training speech, run ``` CUDA_VISIBLE_DEVICES=0 python train_speech.py ```

For training Busi, run ``` CUDA_VISIBLE_DEVICES=0 python train.py ```

## Testing
```
cd into DAG-UNet
```

For training speech, run ``` CUDA_VISIBLE_DEVICES=0 python test_speech.py ```

For training Busi, run ``` CUDA_VISIBLE_DEVICES=0 python val.py ```

## Acknowledgement

We appreciate the work of [UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch) ([paper](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_3)) for providing the foundation of our framework.
