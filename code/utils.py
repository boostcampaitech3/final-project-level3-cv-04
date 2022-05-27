# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

from matplotlib.transforms import Bbox
import numpy as np
import requests
import math
import albumentations as A
import cv2
from pycocotools.coco import COCO
import os

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
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def img_rotate(ann_path, img_root, img_id): # root는 폴더
    
    coco = COCO(ann_path)
    img_path = os.path.join(img_root)
    def get_ann(img_path,api_url) -> dict:
        headers = {"secret": "Boostcamp0000"}
        file_dict = {"file": open(img_path  , "rb")}
        response = requests.post(api_url, headers=headers, files=file_dict)
        return response.json()

    def slope(x, y):
        sl = math.sqrt(x**2 + y**2)
        return sl

    def get_degree(annos):

        horizontal_list = []

        for idx, anno in enumerate(annos):
            xlen = anno['points'][1][0] - anno['points'][0][0] # x축 길이 차 
            horizontal_list.append((xlen, idx))

        longest = max(horizontal_list)[1]

        thetaplus = False
        xlen = annos[longest]['points'][1][0] - annos[longest]['points'][0][0]
        ylen = annos[longest]['points'][0][1] - annos[longest]['points'][1][1] # 음수일 수도 있음

        if ylen < 0 :
            thetaplus = True
            ylen = abs(ylen)

        costheta = max(horizontal_list)[0] / slope(xlen, ylen)
        theta = math.acos(costheta)
        degree = round(theta * 57.29,3)

        if thetaplus == True:
            degree = degree
        else:
            degree = -degree
        return degree

    ann_dict = get_ann(img_path, "http://118.222.179.32:30000/ocr/")
    annos = ann_dict['ocr']['word']

    degree = get_degree(annos)
    
    func_list = [
    A.Rotate(p=1.0, limit=[degree,degree],
    border_mode=cv2.BORDER_CONSTANT
    ),
    ]   
    alb_transform = A.Compose(func_list)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # transformed = alb_transform(image=image, keyPoint = keypoint )
    


# def label_accuracy_score(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#     """
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc, iu