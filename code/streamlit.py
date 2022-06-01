import streamlit as st
import pandas as pd
import numpy as np

import requests
import io
import copy
import torch
import dataset
import albumentations as A
import matplotlib.pyplot as plt
import torchvision
import os

from tkinter import image_names
from PIL import ImageOps, Image, ImageDraw, ImageFont
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm


ocr_url = "http://118.222.179.32:30000/ocr/"
id_list = ['ID', 'ID:', '아이디', 'NETWORK', '네트워크']
pw_list = ['PW', 'PW:', '비밀번호','PASSCODE', 'PASSWORD', '패스워드']
key_list = id_list + pw_list + ['WIFI', 'WI-FI', '와이파이']

class WifiDataset_segmentation(Dataset):
    def __init__(self, coco_dict, ann_dict, api_url, image_path, transform = None, preload = True):
        self.transform = transform
        self.img_root = image_path
        self.coco = coco_dict
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
                # img_name = img_name[0]['file_name']
                x,mask_list,_, __ = img_to_focusmask(image_path,ann_dict)
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
            # img_name = meta[0]['file_name']
            x,mask_list,_, __ = img_to_focusmask(image_path,ann_dict)

        if self.transform:
            transformed = self.transform(               # transform 에 ToTensor 포함됨
                image=x,
                mask=y)
            x = transformed['image'].type(torch.FloatTensor)
            y = transformed['mask'].type(torch.LongTensor)
            t_mask_list = []
            for mask in mask_list:
                transformed = self.transform(image=np.array(mask))
                t_mask_list.append(transformed['image'])
            return x,y,meta,t_mask_list

        else:
            t_mask_list = mask_list
            t = torchvision.transforms.ToTensor()
            return t(x),t(y),meta,t_mask_list


def load_image(image_file):
	img = Image.open(image_file)
	return img


def load_ann_data(image, ocr_url):
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": image}
    response = requests.post(ocr_url, headers=headers, files=file_dict)
    return response.json()


def read_img(image, target_h: int = 1000) -> Image:
    # load image, annotation
    image_bytes = image.getvalue()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert('RGBA')  # 이미지 정보에 따라 이미지를 회전
    area = None
    area_resize= None
    box_args_list=[]
    area_list=[]

    ann_dict = load_ann_data(image_bytes, ocr_url)
    
    # resize
    h, w = img.height, img.width

    ratio = target_h/h
    target_w = int(ratio * w)
    img = img.resize((target_w, target_h))

    # draw polygon
    for val in ann_dict['ocr']['word']:
        poly = val['points']
        word_height = abs(poly[0][1]-poly[2][1])
        
        tag_ori = val['orientation']
        tag_text = val["text"]
        
        key_words=['Pass','PASS']
        for key in key_words:
            if key in tag_text:
                area = copy.deepcopy(poly)
                area[0],area[1],area[2],area[3]=[0,area[0][1]-word_height],[w,area[1][1]-word_height],[w,area[2][1]+word_height*2],[0,area[2][1]+word_height*2]
                area_resize = [list(map(lambda x: x*ratio,temp)) for temp in area]
   
        poly_resize = [list(map(lambda x: x*ratio,temp)) for temp in poly]
        box_args_list.append({"poly_resize":poly_resize, "tag_text":tag_text, "tag_ori":tag_ori})
        if area_resize:area_list.append(area_resize)
        
    draw_polygon(img, box_args_list, area_list)
    return img , ann_dict


def draw_polygon(img,box_args_list,area_list):
    """이미지에 폴리곤을 그린다. illegibility의 여부에 따라 라인 색상이 다르다."""

    img_draw = ImageDraw.Draw(img,'RGBA')
    font_path='/opt/ml/final-project-level3-cv-04/tools/fonts/NanumSquareRoundB.ttf'
    font = ImageFont.truetype(font_path,size=20)
    box_color_RGBA  = (0,255,0,255)
    area_color_RGBA  = (255,0,0,255)
    state_color_RGBA = (0,0,255,50)

    for box in box_args_list:
        pts=box['poly_resize']
        tags=box['tag_text']
        img_draw.text((pts[0][0],pts[0][1]-20),tags,(0,0,0),font,align='left')

        for area in area_list:
            img_draw.rectangle((area[0][0],area[0][1],area[2][0],area[2][1]), outline=state_color_RGBA, width = 3)
            if area[0][0]<pts[0][0] and area[0][1]<pts[0][1] and pts[2][0]<area[2][0] and pts[2][1]<area[2][1]:
                img_draw.rectangle((pts[0][0],pts[0][1],pts[2][0],pts[2][1]), outline=area_color_RGBA, width = 3)
            else:
                img_draw.rectangle((pts[0][0],pts[0][1],pts[2][0],pts[2][1]), outline=box_color_RGBA, width = 3)


def img_to_focusmask(image_path:str, ann_data) -> np.array:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image_gray = image.convert('L')
    image = np.array(image)
    image_gray = np.array(image_gray)

    ann_dict = ann_data
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

    coco = dataset.Custom_COCO(coco_dict)

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

    return out,mask_list,image, coco


def convert_box_mask(images,mask_lists, device):
    new_images = []
    for image,mask_list in zip(images,mask_lists):
        c1,c2,c3 = image
        n_c1 = torch.ones(1,c1.shape[0],c1.shape[1]).to(device)
        n_c2 = torch.zeros(1,c1.shape[0],c1.shape[1]).to(device)
        n_c3 = torch.zeros(1,c1.shape[0],c1.shape[1]).to(device)
        for mask in mask_list:
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
        new_image = torch.cat((n_c1,n_c2,n_c3),dim=0)
        new_images.append(new_image)
    new_images = torch.cat(list(map(lambda x:x.unsqueeze(0),new_images)))
    return new_images


def seg_inference(image_path, ann_dict):
    model_path = '/opt/ml/final-project-level3-cv-04/code/model/unet+++/model.pt'
    state_dict_path = '/opt/ml/final-project-level3-cv-04/code/model/unet+++/170_56.pt'

    image = Image.open(image_path)
    model = torch.load(model_path)
    model.load_state_dict(torch.load(state_dict_path))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = A.Compose([
        A.Resize(512,512),
        ToTensorV2()
    ])

    batch_size = 1
    _, mask_list, __, coco_dict = img_to_focusmask(image_path, ann_dict)
    
    test_dataset = WifiDataset_segmentation(coco_dict, ann_dict, ocr_url, image_path, transform=transform,preload=False)

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fn)

    iter_ = iter(test_dataloader)

    for images_idx,(x,y,meta,mask_list) in enumerate(test_dataloader):
        x,y,meta,mask_list = next(iter_)
        model.eval()
        model.to(device)
        images = model(torch.stack(x).to(device))

        new_images = convert_box_mask(images,mask_list, device)


        for idx in range(batch_size):
            out = torch.argmax(images[idx],dim=0)
            out2 = torch.argmax(new_images[idx],dim=0)
            image_meta = meta[idx][0]

            raw_image = Image.open(image_path)
            t = A.Compose([
                A.Resize(image_meta['height'],image_meta['width']),
                ToTensorV2()
            ])
            t2 = torchvision.transforms.Compose(
                [torchvision.transforms.ToPILImage()]
            )

            plt.figure(figsize=(20,20))

            plt.subplot(batch_size,5,idx*5+1)
            plt.imshow(t2(x[idx]))

            plt.subplot(batch_size,5,idx*5+2)
            plt.imshow(t2(out*0.3))

            plt.subplot(batch_size,5,idx*5+3)
            plt.imshow(t2(out2*0.3))

            plt.subplot(batch_size,5,idx*5+4)
            plt.imshow(t2(y[idx]*0.3))

            plt.subplot(batch_size,5,idx*5+5)
            plt.imshow(raw_image)

        # plt.savefig(f'./out/{images_idx}.jpg')
        st.image(t2(x[idx]))
        st.image(t2(out*0.3))
        st.image(t2(out2*0.3))
        st.image(t2(y[idx]*0.3))
        # st.image(raw_image)


def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())



# main
st.title("Join Wifi")

menu = ["Image","Camera"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Image":
    st.subheader("Upload Wifi Image")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
elif choice == "Camera":
    st.subheader("Take a Wifi Picture")
    image_file = st.camera_input("Take a Picture")


if image_file is not None:

    # To See details
    # file_details = {"filename":image_file.name, "filetype":image_file.type,
    #                 "filesize":image_file.size}
    # st.write(file_details)

    # To View Uploaded Image
    st.image(load_image(image_file),width=250)
    predict_button = st.button('Predict')

    if predict_button:
        # Save Img
        save_dir = './save_images/'
        save_uploaded_file(save_dir, image_file)

        # To Anno tool
        image, ann_dict = read_img(image_file)

        st.write(ann_dict)
        st.image(image, caption='Uploaded Image')

        # seg inference
        image_path = save_dir + image_file.name
        st.write(image_path)
        res_image = seg_inference(image_path, ann_dict)


    # 이미지 확인 => 버튼
    #             => 크롭
    #                     => inference
    #                                  => 결과