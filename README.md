# DAG-UNet

## Data

* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the same folder path as the code for training.
* BUSI Data - The data is available at [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). Create a new folder named `inputs`, and place the downloaded data inside it. For images with multiple masks, combine the masks by adding them together. The processed data can be downloaded from [Google Drive](https://drive.google.com/file/d/1yZKVRyisktEUbcv7msC2L2LELwjRPHAj/view?usp=sharing).

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

For testing speech, run ``` CUDA_VISIBLE_DEVICES=0 python test_speech.py ```

For testing BUSI, run ``` CUDA_VISIBLE_DEVICES=0 python val.py ```

## Testing with trained models

### Trained model
The `./models_pth/speech` directory contains trained models from the five-fold training on speech data

The `./models_pth/busi` directory includes trained models for the BUSI dataset."

### Testing
Run ``` CUDA_VISIBLE_DEVICES=0 python val.py ```

## Acknowledgement

We appreciate the work of [UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch)([paper](https://arxiv.org/abs/2203.04967)) and [Ulite](https://github.com/duong-db/U-Lite) ([paper](https://ieeexplore.ieee.org/document/10317244)) for providing the foundation of our framework.
