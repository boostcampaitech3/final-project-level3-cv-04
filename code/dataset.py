import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np

from PIL import Image, ImageOps
from pycocotools.coco import COCO
import os
from tqdm import tqdm

import utils
from utils import img_to_focusmask, get_ocr, ocr_to_coco, coco_to_mask

id_list = ['ID', '아이디', 'NETWORK', '네트워크', 'IP', 'WIFI']
id_list += list(map(lambda x:x + ':', id_list))
id_list += list(map(lambda x:x + '_', id_list))
pw_list = ['PW', '비밀번호','PASSCODE', 'PASSWORD', '패스워드', 'PIN', 'P.W', '비번']
pw_list += list(map(lambda x:x + ':', pw_list))
pw_list += list(map(lambda x:x + '_', pw_list))
wifi_list = ['WIFI', 'WI-FI', '와이파이', ':', '/']
key_list = id_list + pw_list + wifi_list


class WifiDataset_segmentation(Dataset):
    def __init__(self,ann_path,api_url,image_root,transform,mode = "train") -> None:
        self.transfrom = transform
        self.mode = mode
        self.image_root = image_root
        ####### image_root+ann_path --> image_info_list, ann_list,GT_list
        coco = COCO(ann_path)
        self.img_infos = []
        self.anns = []
        for img_id in coco.getImgIds():
            self.img_infos.append(coco.loadImgs(img_id))
            self.anns.append([])
            for ann_id in coco.getAnnIds(img_id):
                self.anns[-1].append(coco.loadAnns(ann_id))

        ####### image_info_list+api_url --> numpy_image_list + ocr_result
        self.x_list = []            # [img1,img2, ...]
        self.y_list = []            # [mask1,mask2, ...]
        self.ocr_lists = []
        self.c_list = []

        print('load images ...')
        for img_info,ann_list in tqdm(zip(self.img_infos,self.anns),total=len(self.img_infos)):
            img_name = img_info[0]['file_name']
            # get PIL image & ocr out
            image = Image.open(os.path.join(image_root,img_name))
            image = ImageOps.exif_transpose(image).convert('L')
            ocr_out = get_ocr(image,api_url)
            # ann --> GT_mask
            y = np.zeros((image.size[1],image.size[0]))
            for ann in ann_list:
                y[coco.annToMask(ann[0]) == 1] = ann[0]['category_id']

            # image preprocess (PIL --> numpy)
            image, y = utils.img_rotate(image,y)

            ###### ocr_list to mask ---> concat to image c4:all text, c5: wifi key, c6: id key, c7: pw key
            ocr_coco = ocr_to_coco(ocr_out,os.path.join(self.image_root,img_name),(image.shape[0],image.shape[1]))
            c2 = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=False)
            c3 = coco_to_mask(ocr_coco,image.shape,key_list=key_list,get_each_mask=False)

            if mode == 'test':
                _,ocr_out = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=True)

            t = torchvision.transforms.ToPILImage()
            c2 = np.array(t(c2))
            c3 = np.array(t(c3))
            
            self.x_list.append(image)
            self.ocr_lists.append(ocr_out)
            self.y_list.append(y)
            self.c_list.append((c2,c3))

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, idx):
        image = self.x_list[idx]
        y = self.y_list[idx]
        image_info = self.img_infos[idx][0]
        ocr_list = self.ocr_lists[idx]
        c2,c3 = self.c_list[idx]
        if self.mode == 'test':
            ocr_list = self.ocr_lists[idx]
        else:
            ocr_list = None

        # transform에서 tensor로 전환 (C,W,H)
        transformed = self.transfrom(image=np.array(image),mask=y,mask2=c2,mask3=c3)
        x = transformed['image'].type(torch.FloatTensor)
        y = transformed['mask'].type(torch.LongTensor)
        c2 = transformed['mask2']
        c3 = transformed['mask3']
        # concatenate image , masks
        x = torch.cat((x,c2.unsqueeze(0),c3.unsqueeze(0)),dim=0)

        if self.mode == 'test':
            t_ocr_list = []
            for mask,texts in ocr_list:
                transformed = self.transfrom(image=np.array(mask))
                t_ocr_list.append((transformed['image'],texts))

            return x,y,image_info,t_ocr_list

        return x,y,image_info,ocr_list


class five_channel_wifi_dataset(Dataset):
    def __init__(self,ann_path,api_url,image_root,transform,mode = "train") -> None:
        self.transfrom = transform
        self.mode = mode
        self.image_root = image_root
        ####### image_root+ann_path --> image_info_list, ann_list,GT_list
        coco = COCO(ann_path)
        self.img_infos = []
        self.anns = []
        for img_id in coco.getImgIds():
            self.img_infos.append(coco.loadImgs(img_id))
            self.anns.append([])
            for ann_id in coco.getAnnIds(img_id):
                self.anns[-1].append(coco.loadAnns(ann_id))

        ####### image_info_list+api_url --> numpy_image_list + ocr_result
        self.x_list = []            # [img1,img2, ...]
        self.y_list = []            # [mask1,mask2, ...]
        self.ocr_lists = []
        self.c_list = []

        print('load images ...')
        for img_info,ann_list in tqdm(zip(self.img_infos,self.anns),total=len(self.img_infos)):
            img_name = img_info[0]['file_name']
            # get PIL image & ocr out
            image = Image.open(os.path.join(image_root,img_name))
            image = ImageOps.exif_transpose(image).convert('RGB')
            ocr_out = get_ocr(image,api_url)
            # ann --> GT_mask
            y = np.zeros((image.size[1],image.size[0]))
            for ann in ann_list:
                y[coco.annToMask(ann[0]) == 1] = ann[0]['category_id']

            ###### ocr_list to mask ---> concat to image 24:all text, c5:3wifi key, c6:image = np.array(image)
            ocr_coco = ocr_to_coco(ocr_out,os.path.join(self.image_root,img_name),(image.shape[0],image.shape[1]))
            c4 = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=False)
            c5 = coco_to_mask(ocr_coco,image.shape,key_list=wifi_list,get_each_mask=False)
            c6 = coco_to_mask(ocr_coco,image.shape,key_list=id_list,get_each_mask=False)
            c7 = coco_to_mask(ocr_coco,image.shape,key_list=pw_list,get_each_mask=False)      # output: torch.tensor
            if mode == 'test':
                _,ocr_out = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=True)

            t = torchvision.transforms.ToPILImage()
            c4 = np.array(t(c4))
            c5 = np.array(t(c5))
            c6 = np.array(t(c6))
            c7 = np.array(t(c7))
            
            self.x_list.append(image)
            self.ocr_lists.append(ocr_out)
            self.y_list.append(y)
            self.c_list.append((c4,c5,c6,c7))

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, idx):
        image = self.x_list[idx]
        y = self.y_list[idx]
        image_info = self.img_infos[idx][0]
        ocr_list = self.ocr_lists[idx]
        c4,c5,c6,c7 = self.c_list[idx]
        if self.mode == 'test':
            ocr_list = self.ocr_lists[idx]
        else:
            ocr_list = None

        # transform에서 tensor로 전환 (C,W,H)
        transformed = self.transfrom(image=np.array(image),mask=y,mask4=c4,mask5=c5,mask6=c6,mask7=c7)
        x = transformed['image'].type(torch.FloatTensor)
        y = transformed['mask'].type(torch.LongTensor)
        c4 = transformed['mask4']
        c5 = transformed['mask5']
        c6 = transformed['mask6']
        c7 = transformed['mask7']
        # concatenate image , masks
        x = torch.cat((x,c4.unsqueeze(0),c5.unsqueeze(0),c6.unsqueeze(0),c7.unsqueeze(0)),dim=0)

        if self.mode == 'test':
            t_ocr_list = []
            for mask,texts in ocr_list:
                transformed = self.transfrom(image=np.array(mask))
                t_ocr_list.append((transformed['image'],texts))

            return x,y,image_info,t_ocr_list

        return x,y,image_info,ocr_list


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