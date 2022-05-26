import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np

from pycocotools.coco import COCO
import os
from tqdm import tqdm

from utils import img_to_focusmask

id_list = ['ID', 'ID:', '아이디', 'NETWORK', '네트워크']
pw_list = ['PW', 'PW:', '비밀번호','PASSCODE', 'PASSWORD', '패스워드']
key_list = id_list + pw_list + ['WIFI', 'WI-FI', '와이파이']


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
            for mask,texts in mask_list:
                transformed = self.transfrom(image=np.array(mask))
                t_mask_list.append((transformed['image'],texts))
            return x,y,meta,t_mask_list

        else:
            t = torchvision.transforms.ToTensor()
            return t(x),t(y),meta,mask_list


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