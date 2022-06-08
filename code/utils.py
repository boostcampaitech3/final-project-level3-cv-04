import requests
import time
import json
import io

import numpy as np

import requests
import math
import albumentations as A
import cv2
import os
import imutils
import torch
import torchvision

from PIL import Image, ImageOps
from pycocotools.coco import COCO
from collections import defaultdict


class Custom_COCO(COCO):
    def __init__(self, annotation_file=None):
        """
        annotation_file : path or dict
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


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def slope(x, y):
    sl = math.sqrt(x**2 + y**2)
    return sl


def get_degree(annos):

    # 제일 대표적인 bbox를 찾음
    horizontal_list = []

    for idx, anno in enumerate(annos):
        xlen = anno['points'][1][0] - anno['points'][0][0] # x축 길이 차
        ylen = anno['points'][0][1] - anno['points'][1][1] # y축 길이 차
        ylen = abs(ylen)
        xlen2 = anno['points'][2][0] - anno['points'][1][0] # x축 길이 차
        xlen2 = abs(xlen2)
        ylen2 = anno['points'][1][1] - anno['points'][2][1] # y축 길이 차
        ylen2 = abs(ylen2)
        horizontal_list.append((slope(xlen,ylen)/slope(xlen2,ylen2), idx, anno["text"],xlen,ylen))

    longest = max(horizontal_list)[1]

    # 각도 계산
    thetaplus = False
    xlen = annos[longest]['points'][1][0] - annos[longest]['points'][0][0]
    ylen = annos[longest]['points'][0][1] - annos[longest]['points'][1][1] # 음수일 수도 있음

    if ylen < 0 :
        thetaplus = True
        ylen = abs(ylen)

    costheta = xlen / slope(xlen, ylen)
    theta = math.acos(costheta)
    degree = round(theta * 57.29,1)

    if thetaplus == True:
        degree = degree
    else:
        degree = -degree

    return degree


def img_rotate(image,mask=None):

    ann_dict = get_ocr(image, "http://118.222.179.32:30001/ocr/")
    annos = ann_dict['ocr']['word']

    degree = get_degree(annos)
    
    rotated = imutils.rotate_bound(np.array(image), -degree)
    if type(mask) == type(None):
        return rotated
    else:
        r_mask = imutils.rotate_bound(mask, -degree)
        return rotated



def seg_to_classification(image:torch.tensor,mask_list,device):
    out = {'id':[],'pw':[]}
    c1,c2,c3 = image
    n_c1 = torch.ones(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
    n_c2 = torch.zeros(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
    n_c3 = torch.zeros(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
    for mask,text in mask_list:
        mask = mask[0]
        dic = {}
        dic[0] = sum(c1[mask == 1].data)
        dic[1] = sum(c2[mask == 1].data)
        dic[2] = sum(c3[mask == 1].data)
        ratio_dic = {}
        if sum([dic[0],dic[1],dic[2]]) == 0:
            continue
        ratio_dic[0] = dic[0]/sum([dic[0],dic[1],dic[2]])
        ratio_dic[1] = dic[1]/sum([dic[0],dic[1],dic[2]])
        ratio_dic[2] = dic[2]/sum([dic[0],dic[1],dic[2]])
        n_c1[0][mask == 1] = ratio_dic[0]
        n_c2[0][mask == 1] = ratio_dic[1]
        n_c3[0][mask == 1] = ratio_dic[2]
        if sorted(ratio_dic.items(), key = lambda item: item[1])[-1][0] == 1:
            out['id'].append(text)
        elif sorted(ratio_dic.items(), key = lambda item: item[1])[-1][0] == 2:
            out['pw'].append(text)
    new_image = torch.cat((n_c1,n_c2,n_c3),dim=0)

    return new_image,out

def seg_to_boxmask(images:torch.tensor,mask_lists,device) -> torch.tensor:
    '''
    image segment --> bbox image, {id: ..., pw: ...}
    '''
    new_images = []
    out_list = []
    for image,mask_list in zip(images,mask_lists):
        out_list.append({'id':[],'pw':[]})
        c1,c2,c3 = image
        n_c1 = torch.ones(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
        n_c2 = torch.zeros(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
        n_c3 = torch.zeros(1,c1.shape[0],c1.shape[1],requires_grad=True).to(device)
        for mask,text in mask_list:
            mask = mask[0]
            dic = {}
            dic[0] = sum(c1[mask == 1].data)
            dic[1] = sum(c2[mask == 1].data)
            dic[2] = sum(c3[mask == 1].data)
            ratio_dic = {}
            if sum([dic[0],dic[1],dic[2]]) == 0:
                continue
            ratio_dic[0] = dic[0]/sum([dic[0],dic[1],dic[2]])
            ratio_dic[1] = dic[1]/sum([dic[0],dic[1],dic[2]])
            ratio_dic[2] = dic[2]/sum([dic[0],dic[1],dic[2]])
            n_c1[0][mask == 1] = ratio_dic[0]
            n_c2[0][mask == 1] = ratio_dic[1]
            n_c3[0][mask == 1] = ratio_dic[2]
            if sorted(ratio_dic.items(), key = lambda item: item[1])[-1][0] == 1:
                out_list[-1]['id'].append(text)
            elif sorted(ratio_dic.items(), key = lambda item: item[1])[-1][0] == 2:
                out_list[-1]['pw'].append(text)
        new_image = torch.cat((n_c1,n_c2,n_c3),dim=0)
        new_images.append(new_image)
    new_images = torch.cat(list(map(lambda x:x.unsqueeze(0),new_images)))
    return new_images,out_list


def get_ocr(img_path,api_url:str) -> dict:
    ''' img_path : str or PIL image '''
    if type(img_path) == str:
        file_dict = {"file": open(img_path  , "rb")}
    else:
        output = io.BytesIO()
        image = img_path
        image.save(output, format="JPEG")
        file_dict = {"file": output.getvalue()}

    headers = {"secret": "Boostcamp0001"}
    response = requests.post(api_url, headers=headers, files=file_dict)


    return response.json()


def ocr_to_coco(ocr_result,image_path,image_shape:tuple) -> COCO:
    coco_dict = {
        'images':[{'filename':image_path,'height': image_shape[0],'width': image_shape[1],'id': 0}],
        'categories':[{'id':1,"name": "text"}],
        'annotations':[]
    }

    for i,box in enumerate(ocr_result['ocr']['word']):
        coco_dict['annotations'].append({
            'id':i,
            'image_id':0,
            'category_id':1,
            'text': box['text'],
            'segmentation': [[]]
        })
        for point in box['points']:
            coco_dict['annotations'][-1]['segmentation'][0] += point

    return Custom_COCO(coco_dict)


def coco_to_mask(coco,image_size:tuple,key_list =None,get_each_mask=True) -> torch.tensor:
    c1 = torch.zeros((1,image_size[0], image_size[1]))
    for ann in coco.anns.values():
        if key_list:
            if ann['text'].upper() in key_list:
                c1[0][coco.annToMask(ann) == 1] = 255
        else:
            c1[0][coco.annToMask(ann) == 1] = 255
    if get_each_mask:
        mask_list = []
        for ann in coco.anns.values():
            mask_list.append((coco.annToMask(ann),ann['text']))
        return c1,mask_list
    return c1


def img_to_focusmask(image_path:str,api_url:str,key_list) -> np.array:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image_gray = image.convert('L')
    image = np.array(image)
    image_gray = np.array(image_gray)

    ann_dict = get_ocr(image_path,api_url)

    coco = ocr_to_coco(ann_dict,image_path,image.shape)

    c1,mask_list = coco_to_mask(coco,image.shape,key_list=None,get_each_mask=True)
    c2 = coco_to_mask(coco,image.shape,key_list=key_list,get_each_mask=False)

    t = torchvision.transforms.ToPILImage()
    c1 = np.array(t(c1))
    c2 = np.array(t(c2))
    c1 = np.reshape(c1,(c1.shape[0],c1.shape[1],1))
    c2 = np.reshape(c2,(c1.shape[0],c1.shape[1],1))
    image_gray = np.reshape(image_gray,(c1.shape[0],c1.shape[1],1))
    out = np.concatenate((image_gray,c1,c2),axis=2)

    return out,mask_list,image