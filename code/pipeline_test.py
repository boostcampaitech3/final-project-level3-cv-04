from unicodedata import name
import utils
import dataset
import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pipeline


if __name__ == '__main__':

    folder_path = '/opt/ml/upstage_OCR/Data set/real data/general'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    out_path = './out/test/general'
    model = torch.load('/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0+gen+receipt/model.pt')
    model.load_state_dict(torch.load('/opt/ml/upstage_OCR/code/saved/c1_k0/480.pt'))

    imagelist = sorted(os.listdir(folder_path))
    for path in imagelist:
        imgpath = os.path.join(folder_path, path)
        print(path)
        pipeline.pipeline(imgpath,model,device,out_path,path.split('.')[0])
        print("=========================================================================================")
