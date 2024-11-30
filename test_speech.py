import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.utils import MetricReduction
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.data import (
    Dataset)
from net import DAG_UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

no_classes = 7

no_epochs = 200
k_folds = 5

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hausdorff_metric = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=None, directed=False, reduction=MetricReduction.MEAN, get_not_nans=False)
classes = [1,2,3,4,5,6]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPERIMENT_CODE = 1


model = DAG_UNet(in_c=1,out_c=7,threshold=0.6,frac=0.8).cpu()
experiment_number = EXPERIMENT_CODE
metric_values = list()
metrics_list = []
tes=['aa','ah','br','gc','mr']
df=pd.DataFrame(columns=['Volunteer_test','Frame','Class','Dice','HD','Model'])
for i in range(k_folds):
    test_vol=tes[i]
    state_dict = torch.load(f'./model_pth/speech/best_metric_model_{i}.pth')
    # Filter out unwanted keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}

    model.load_state_dict(filtered_state_dict)
    
    model.to(device)
    model.eval()
    test_set = torch.load(f"./test_dataset/test_{i+1}_1.pt", map_location=torch.device('cpu'))
    testloaderCV = torch.utils.data.DataLoader(test_set, shuffle=False)
    metrics = []
    outputs = np.zeros([1,7,256,256])
    k=0
    for test_data in testloaderCV:
        k+=1
        test_images, test_labels = test_data['img'].to(device), test_data['seg'].to(device)
 
        test_outputs  = model(test_images)
        test=test_outputs
        test_outputs = torch.argmax(test, dim=1)
        test_outputs = F.one_hot(test_outputs, num_classes = -1)
        test_outputs = torch.permute(test_outputs, (0, 3, 1, 2))
        
        test_labels = F.one_hot(test_labels, num_classes = no_classes)
        test_labels = torch.permute(test_labels, (0, 1, 4, 2, 3))
        test_labels = torch.squeeze(test_labels, dim=1)
        dice_metric(y_pred=test_outputs, y=test_labels)
        pp=dice_metric(y_pred=test_outputs, y=test_labels)
        m=torch.mean(pp,0,True)
        pp_HD=hausdorff_metric(y_pred=test_outputs, y=test_labels)
        
        if test_data == 0:
            outputs = test_outputs.detach().cpu().numpy()
            print(str(np.shape(outputs)) + 'outputs 1')
        else:
            outputs = np.append(outputs, test_outputs.detach().cpu().numpy(), axis = 0)
        
                    
            result=[
                    [test_vol,k,'Head',float(m[0,1]),float(pp_HD[0,1]),'DAG-UNet'],
                    [test_vol,k,'Soft-palate',float(m[0,2]),float(pp_HD[0,2]),'DAG-UNet'],
                    [test_vol,k,'Jaw',float(m[0,3]),float(pp_HD[0,3]),'DAG-UNet'],
                    [test_vol,k,'Tongue',float(m[0,4]),float(pp_HD[0,4]),'DAG-UNet'],
                    [test_vol,k,'Vocal-Tract',float(m[0,5]),float(pp_HD[0,5]),'DAG-UNet'],
                    [test_vol,k,'Tooth-space',float(m[0,6]),float(pp_HD[0,6]),'DAG-UNet']
                    ]
            
            df1=pd.DataFrame(result,columns=df.columns)
            df=df.append(df1)


    print(str(np.shape(outputs)) + 'outputs at end')

    metric = dice_metric.aggregate().item()
    metric1 = dice_metric
    print(metric)
    path="./pred"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"The new directory {path} is created!")
    np.save(path  + f'/Sub_{i+1}_outputs', outputs)
    dice_metric.reset()
    print(type(metric1))
    metric_values.append(metric)
    

del model
print(df.shape)
df.to_csv('./pred/test_DAG-UNet_.csv',index=True)   

              
