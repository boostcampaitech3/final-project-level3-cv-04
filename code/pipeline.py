from unicodedata import name
import utils
import dataset

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    return


if __name__ == '__main__':
    
    img_path = '/opt/ml/upstage_OCR/Data set/real data/general/general003.jpg'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/model.pt')
    model.load_state_dict(torch.load('/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/420_73.9.pt'))

    pipeline(img_path,model,device)