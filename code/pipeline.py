from unicodedata import name
import utils
import dataset
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

def bbox_concat(bbox_list):
    texts = []
    for ind, anno in enumerate(bbox_list):
        texts.append((ind, anno[1][0],anno[1][1], anno[0], anno[1][2],anno[1][3],)) 

    texts_ = sorted(texts, key = lambda x: (x[1][1],x[1][0]))  # y로 정렬 후 x정렬

    tmp = texts_[0][1][1] # 첫번째 글자의 y좌표 
    align = []
    phase = []
    for text in texts_:
        new_phase = []
        if abs(text[1][1] - tmp) <= ((text[5][1] - text[1][1])/3) :  #  같은 라인 판별 / 글자의 반 이내면 
            phase.append(text)
            tmp = text[1][1]
    else: # tmp값 벌어지면 다음 라인 취급
        phase.sort(key = lambda x: x[1][0]) # x로 정렬
        align.append(phase)
        new_phase.append(text)
        phase = new_phase
        tmp = text[2][1]

    phase.sort(key = lambda x: x[1][0])
    align.append(phase) # 마지막 줄 추가


    print("--------------------")
    line = []
    word = []
    for i in align:
        tmp = i[0][1][0]
        for n in i:
            if n[1][0] - tmp <= 100: 
                word.append(n[3])
                tmp = n[2][0]
            else:
                word.append(" ")
                word.append(n[3])
                tmp = n[2][0]
        line.append(word)
        word = []

    for i in line:
        s = "".join(i)
        print(s)

    print(line)


def get_3chanel_key_masked_image(image,ocr,img_path):
    ocr_coco = utils.ocr_to_coco(ocr,img_path,(image.shape[0],image.shape[1]))
    c2 = utils.coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=False)
    c3 = utils.coco_to_mask(ocr_coco,image.shape,key_list=dataset.key_list,get_each_mask=False)
    _,mask_list = utils.coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=True)
    return torch.cat((torch.tensor(image).unsqueeze(0),c2,c3),dim=0), mask_list


def pipeline(img_path,model,device):
    ### 1. img_path --> PIL image ###
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image).convert('L')      #image:PIL
    
    ### 2. PIL image rotate ###
    image = utils.img_rotate(image)             #image:np.array

    ### 3. get ocr ###
    ocr = utils.get_ocr(Image.fromarray(image),"http://118.222.179.32:30000/ocr/")

    ### 4. get key masked image, each mask list ###
    x, mask_list = get_3chanel_key_masked_image(image,ocr,img_path)     # x:torch.tensor
    PIL_transform =torchvision.transforms.ToPILImage()
    x = PIL_transform(x)                                                # x:PIL image

    ### 5. use model: key masked image --> segmentation map ###
    t = A.Compose([
                A.Resize(512,512),
                ToTensorV2()
            ])
    x = t(image = np.array(x))['image'].type(torch.FloatTensor)         # x:torch.tensor
    t_ocr_list = []
    for (mask,texts),location in zip(mask_list,ocr['ocr']['word']):
        transformed = t(image=np.array(mask))
        t_ocr_list.append((transformed['image'],(texts,location['points'])))
    model.to(device)
    pred = model(x.unsqueeze(0).to(device))[0]

    ### 6. segmentation map + mask_list --> id, pw value classification list ###
    classificated_image,out_list = utils.seg_to_classification(pred,t_ocr_list,device)

    ### save images ##
    out = torch.argmax(pred,dim=0)
    out2 = torch.argmax(classificated_image,dim=0)
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.subplot(1,4,2)
    plt.imshow(PIL_transform(x))
    plt.subplot(1,4,3)
    plt.imshow(PIL_transform(out*0.3))
    plt.subplot(1,4,4)
    plt.imshow(PIL_transform(out2*0.3))
    plt.savefig(f'./out.jpg')

    print(out_list)
    ### TODO post processing: id, pw value list
    if out_list['id']:
        bbox_concat(out_list['id'])
    if out_list['pw']:
        bbox_concat(out_list['pw'])
    return


if __name__ == '__main__':


    folder_path = '/opt/ml/upstage_OCR/Data set/real data/general'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/model.pt')
    model.load_state_dict(torch.load('/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/420_73.9.pt'))

    imagelist = sorted(os.listdir(folder_path))
    for path in imagelist:
        imgpath = os.path.join(folder_path, path)
    
        pipeline(imgpath,model,device)
        print("=========================================================================================")
        