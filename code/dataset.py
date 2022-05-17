import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import requests
from pycocotools.coco import COCO
from PIL import Image, ImageOps
import os
from tqdm import tqdm


id_list = ['ID', 'ID:', '아이디', 'NETWORK', '네트워크']
pw_list = ['PW', 'PW:', '비밀번호','PASSCODE', 'PASSWORD', '패스워드']
key_list = id_list + pw_list + [':','WIFI', '1층', '2층', '3층', '4층', 'FREE', 'WI-FI', 'KT', 'GIGA','와이파이']


def get_ann(img_path:str,api_url:str) -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": open(img_path  , "rb")}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()


def img_to_focusmask(image_path:str,api_url:str) -> torch.Tensor:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

    image = torchvision.transforms.ToTensor()(image)
    image_gray = torchvision.transforms.Grayscale()(image)

    ann_dict = get_ann(image_path,api_url)
    coco_dict = {
        'images':[{
            'filename':image_path,
            'height': image.shape[1],
            'width': image.shape[2],
            'id': 0
        }],
        'categories':[
            {
                'id':1,
                "name": "text"
            }
        ],
        'annotations':[]
    }

    for i,box in enumerate(ann_dict['ocr']['word']):
        coco_dict['annotations'].append({
            'id':i,
            'image_id':0,
            'category_id':1,
            'text': box['text'],
            'segmentation': [[]]
        })
        for point in box['points']:
            coco_dict['annotations'][-1]['segmentation'][0] += point

    coco = COCO(coco_dict)

    c1 = np.zeros((1,image.shape[1], image.shape[2]))
    c2 = np.zeros((1,image.shape[1], image.shape[2]))

    for ann in coco.anns.values():
        c1[0][coco.annToMask(ann) == 1] = ann['category_id']

    for ann in coco.anns.values():
        if any(map(lambda x: x in ann['text'],key_list)):
            c2[0][coco.annToMask(ann) == 1] = ann['category_id']

    c1 = torch.from_numpy(c1)
    c2 = torch.from_numpy(c2)
    out = torch.cat((image_gray,c1,c2))

    return out


class WifiDataset_segmentation(Dataset):
    def __init__(self,ann_path,api_url,image_root,device):
        self.img_root = image_root
        self.coco = COCO(ann_path)
        self.api_url = api_url
        self.img_names = []
        self.anns = []
        for img_id in self.coco.getImgIds():
            self.img_names.append(self.coco.loadImgs(img_id))
            self.anns.append([])
            for ann_id in self.coco.getAnnIds(img_id):
                self.anns[-1].append(self.coco.loadAnns(ann_id))

        self.x_list = []
        self.y_list = []
        print('load images ...')
        for img_name,ann_list in tqdm(zip(self.img_names,self.anns),total=len(self.img_names)):
            img_name = img_name[0]['file_name']
            x = img_to_focusmask(os.path.join(self.img_root,img_name),self.api_url)
            y = torch.zeros((1,x.shape[1],x.shape[2]))
            for ann in ann_list:
                y[0][self.coco.annToMask(ann[0]) == 1] = ann[0]['category_id']
            self.x_list.append(x.to(device))
            self.y_list.append(y.to(device))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]