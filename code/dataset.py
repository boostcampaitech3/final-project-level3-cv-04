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
key_list = id_list + pw_list + ['WIFI', 'WI-FI', '와이파이']


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


def img_to_focusmask(image_path:str,api_url:str) -> np.array:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image_gray = image.convert('L')
    image = np.array(image)
    image_gray = np.array(image_gray)

    ann_dict = get_ann(image_path,api_url)
    coco_dict = {
        'images':[{'filename':image_path,'height': image.shape[0],'width': image.shape[1],'id': 0}],
        'categories':[{'id':1,"name": "text"}],
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

    c1 = torch.zeros((1,image.shape[0], image.shape[1]))
    c2 = torch.zeros((1,image.shape[0], image.shape[1]))

    mask_list = []
    for ann in coco.anns.values():
        c1[0][coco.annToMask(ann) == 1] = 255
        mask_list.append(coco.annToMask(ann))

    for ann in coco.anns.values():
        if any(map(lambda x: x in ann['text'],key_list)):
            c2[0][coco.annToMask(ann) == 1] = 255

    t = torchvision.transforms.ToPILImage()
    c1 = np.array(t(c1))
    c2 = np.array(t(c2))
    c1 = np.reshape(c1,(c1.shape[0],c1.shape[1],1))
    c2 = np.reshape(c2,(c1.shape[0],c1.shape[1],1))
    image_gray = np.reshape(image_gray,(c1.shape[0],c1.shape[1],1))
    out = np.concatenate((image_gray,c1,c2),axis=2)

    return out,mask_list,image


class WifiDataset_segmentation(Dataset):
    def __init__(self,ann_path,api_url,image_root,transform = None, preload = True):
        self.transfrom = transform
        self.img_root = image_root
        self.coco = COCO(ann_path)
        self.api_url = api_url
        self.preload = preload
        self.img_metas = []             # 이미지별 메타 정보 list
        self.anns = []                  # [[ann1-1, ann1-2], [ann2-1,ann2-2], ... ]
        for img_id in self.coco.getImgIds():
            self.img_metas.append(self.coco.loadImgs(img_id))
            self.anns.append([])
            for ann_id in self.coco.getAnnIds(img_id):
                self.anns[-1].append(self.coco.loadAnns(ann_id))

        if preload == True:
            self.x_list = []            # [img1,img2, ...]
            self.y_list = []            # [mask1,mask2, ...]
            self.mask_lists = []
            
            print('load images ...')
            for img_name,ann_list in tqdm(zip(self.img_metas,self.anns),total=len(self.img_metas)):
                img_name = img_name[0]['file_name']
                x,mask_list,_ = img_to_focusmask(os.path.join(self.img_root,img_name),self.api_url)
                self.mask_lists.append(mask_list)
                y = np.zeros((x.shape[0],x.shape[1]))
                for ann in ann_list:
                    y[self.coco.annToMask(ann[0]) == 1] = ann[0]['category_id']

                self.x_list.append(x)
                self.y_list.append(y)

    def __len__(self):
        return len(self.img_metas)
    
    def __getitem__(self, idx):
        meta = self.img_metas[idx]
        ann_list = self.anns[idx]
        y = np.zeros((meta[0]['height'],meta[0]['width']))
        for ann in ann_list:
            y[self.coco.annToMask(ann[0]) == 1] = ann[0]['category_id']
        
        if self.preload:
            x = self.x_list[idx]
            mask_list = self.mask_lists[idx]
        else:
            img_name = meta[0]['file_name']
            x,mask_list,_ = img_to_focusmask(os.path.join(self.img_root,img_name),self.api_url)

        if self.transfrom:
            transformed = self.transfrom(               # transform 에 ToTensor 포함됨
                image=x,
                mask=y)
            x = transformed['image'].type(torch.FloatTensor)
            y = transformed['mask'].type(torch.LongTensor)
            t_mask_list = []
            for mask in mask_list:
                transformed = self.transfrom(image=np.array(mask))
                t_mask_list.append(transformed['image'])
            return x,y,meta,t_mask_list

        else:
            t_mask_list = mask_list
            t = torchvision.transforms.ToTensor()
            return t(x),t(y),meta,t_mask_list


class Concat_Dataset(Dataset):
    def __init__(self,dataset_list):
        super().__init__()
        self.dataset_list = dataset_list
        self.len_list = []
        for dataset in self.dataset_list:
            self.len_list.append(len(dataset))


    def __getitem__(self, index): 
        for i,l in enumerate(self.len_list):
            if index < l:
                return self.dataset_list[i][index]
            else:
                index -= l

    def __len__(self):
        return sum(self.len_list)