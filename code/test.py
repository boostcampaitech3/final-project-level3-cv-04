import torch
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torchvision
import os
from PIL import Image

model_path = '/opt/ml/upstage_OCR/code/saved/unet+++/model.pt'
state_dict_path = '/opt/ml/upstage_OCR/code/saved/unet+++/170_56.pt'

model = torch.load(model_path)
model.load_state_dict(torch.load(state_dict_path))

ann_path = '/opt/ml/upstage_OCR/Data set/valid_general.json'
ocr_url = "http://118.222.179.32:30000/ocr/"
image_root = '/opt/ml/upstage_OCR/Data set/real data/general'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = A.Compose([
    A.Resize(512,512),
    ToTensorV2()
])

batch_size = 1

test_dataset = dataset.WifiDataset_segmentation(ann_path,ocr_url,image_root,transform=transform,preload=False)

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)

def convert_box_mask(images,mask_lists):
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

iter_ = iter(test_dataloader)

for images_idx,(x,y,meta,mask_list) in enumerate(test_dataloader):
    x,y,meta,mask_list = next(iter_)
    model.eval()
    model.to(device)
    images = model(torch.stack(x).to(device))

    new_images = convert_box_mask(images,mask_list)


    for idx in range(batch_size):
        out = torch.argmax(images[idx],dim=0)
        out2 = torch.argmax(new_images[idx],dim=0)
        image_meta = meta[idx][0]

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

    plt.savefig(f'./out/{images_idx}.jpg')