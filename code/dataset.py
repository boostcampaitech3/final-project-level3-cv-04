import json
import time
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import requests
from pycocotools.coco import COCO
from PIL import Image, ImageOps
import os
from tqdm import tqdm
from collections import defaultdict


id_list = ['ID', 'ID:', '아이디', 'NETWORK', '네트워크']
pw_list = ['PW', 'PW:', '비밀번호','PASSCODE', 'PASSWORD', '패스워드']
key_list = id_list + pw_list + [':','WIFI', '1층', '2층', '3층', '4층', 'FREE', 'WI-FI', 'KT', 'GIGA','와이파이']


class Custom_COCO(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if type(annotation_file) == str:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

        elif type(annotation_file) == dict:
            tic = time.time()
            self.dataset = annotation_file
            self.createIndex()


    def createIndex(self):
        # create index
        # print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats


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

    coco = Custom_COCO(coco_dict)

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