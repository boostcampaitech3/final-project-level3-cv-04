import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import requests
from pycocotools.coco import COCO
from PIL import Image, ImageOps


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
            print('key :', ann['text'])
        else:
            print('passed :', ann['text'])
            pass

    c1 = torch.from_numpy(c1)
    c2 = torch.from_numpy(c2)
    out = torch.cat((image_gray,c1,c2))

    return out



class WifiDataset(Dataset):
    def __init__(self,ann_path:str,api_url):
        self.coco = COCO(ann_path)
        self.api_url = api_url
        self.img_names = []
        self.anns = []
        for img_id in self.coco.getImgIds():
            self.img_names.append(self.coco.loadImgs(img_id))
            self.anns.append([])
            for ann_id in self.coco.getAnnIds(img_id):
                self.anns[-1].append(self.coco.loadAnns(ann_id))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        ann_list= self.anns[idx]
        x = img_to_focusmask(img_name,self.api_url)

        y = torch.zeros((1,x.shape(1),x.shape(2)))
        for ann in ann_list:
            y[0][self.coco.annToMask(ann[0]) == 1] = ann[0]['category_id']
        return x, y