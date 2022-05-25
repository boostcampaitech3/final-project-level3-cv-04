import os
import torch
from utils import label_accuracy_score, add_hist, convert_box_mask
import numpy as np
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset
import model
import loss


ann_path = '/opt/ml/upstage_OCR/Data set/train_general.json'
ann_path_val = '/opt/ml/upstage_OCR/Data set/valid_general.json'
ocr_url = "http://118.222.179.32:30000/ocr/"
image_root = '/opt/ml/upstage_OCR/Data set/real data/general'
sorted_df = pd.DataFrame({'Categories':['background','ID','PW']})
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 2000
batch_size = 4
val_every = 10
preload = True

saved_dir = './saved/unet+++'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)


# model 정의
# model_ = model.UNet_3Plus(n_classes=3)
model_ = model.UNet_3Plus(n_classes=3)

# Loss function 정의
# criterion = torch.nn.CrossEntropyLoss()
criterion = loss.FocalLoss()

# Optimizer 정의
optimizer = torch.optim.Adam(params = model_.parameters(), lr = 0.0001, weight_decay=1e-6)


transform = A.Compose([
    A.Resize(512,512),
    # A.GaussNoise(var_limit=(50,100)),
    # A.MotionBlur(blur_limit=5),
    A.Blur(5),
    A.OneOf([
        A.ShiftScaleRotate(rotate_limit=(-45,45),p=1),
        # A.ElasticTransform(sigma=30,alpha_affine=30,p=1),
        ],p=0.5),
    ToTensorV2()
    ])

train_dataset = dataset.WifiDataset_segmentation(ann_path,ocr_url,image_root,transform=transform,preload=preload)
val_dataset = dataset.WifiDataset_segmentation(ann_path_val,ocr_url,image_root,transform=transform,preload=preload)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    for epoch in range(num_epochs):
        model.train()
        torch.save(model,f'{saved_dir}/model.pt')
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _, mask_list) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)
            # outputs = convert_box_mask(outputs,mask_list,device)        # segmentation --> box
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
            torch.save(model.state_dict(),f'{saved_dir}/{epoch+1}.pt')

    torch.save(model.state_dict(),f'{saved_dir}/last.pt')


def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()
    
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _, mask_list) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            # outputs = convert_box_mask(outputs,mask_list,device)        # segmentation --> box
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss


train(num_epochs, model_, train_dataloader, val_dataloader, criterion, optimizer, saved_dir, val_every, device)