import os
import json

import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import utils
import dataset

model_path = '/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/model.pt'
state_dict_path = '/opt/ml/upstage_OCR/code/saved/unet++_3c_rotate_k0/420_73.9.pt'

model = torch.load(model_path)
model.load_state_dict(torch.load(state_dict_path))

ann_path = '/opt/ml/upstage_OCR/Data set/test_0.json'
ocr_url = "http://118.222.179.32:30000/ocr/"
image_root = '/opt/ml/upstage_OCR/Data set/real data/general'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
out_folder = 'unet++_3c_rotate_k0'

transform = A.Compose([
    A.LongestMaxSize(max_size=512, interpolation=1),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
    # A.Resize(512,512),
    ToTensorV2()
], additional_targets={'mask2': 'mask', 'mask3': 'mask'})

batch_size = 1

os.makedirs(f'out/{out_folder}')


test_dataset = dataset.WifiDataset_segmentation(ann_path,ocr_url,image_root,transform=transform,mode='test')

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)


for images_idx,(x,y,meta,mask_list) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
    model.eval()
    model.to(device)
    images = model(torch.stack(x).to(device))

    new_images,out_list = utils.seg_to_boxmask(images,mask_list,device)

    for idx in range(batch_size):
        out = torch.argmax(images[idx],dim=0)
        out2 = torch.argmax(new_images[idx],dim=0)
        image_meta = meta[idx]

        raw_image = Image.open(os.path.join(image_root,image_meta['file_name']))
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

    plt.savefig(f'./out/{out_folder}/{images_idx}.jpg')

    with open(f'out/{out_folder}/{images_idx}.json', 'w') as f:
        json.dump(out_list[0],f,indent=4, ensure_ascii=False)