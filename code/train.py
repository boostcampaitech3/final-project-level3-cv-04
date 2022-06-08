import os
import argparse
import yaml
import torch
from utils import label_accuracy_score, add_hist
import numpy as np
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset
import model
import loss

class main():

    def __init__(self,main_config):
        self.train_json_path_list = main_config['train_json_path_list']
        self.val_json_path_list = main_config['val_json_path_list']
        self.train_image_path_list = main_config['train_image_path_list']
        self.val_image_path_list = main_config['val_image_path_list']
        self.save_path = main_config['save_path']
        self.epochs = main_config['epochs']
        self.batch_size = main_config['batch_size']
        self.validate = main_config['validate']
        self.eval_interval = main_config['eval_interval']
        self.lr = main_config['lr']
        ocr_url = "http://118.222.179.32:30001/ocr/"
        
        self.model = model.UNetPlusPlus(out_ch=3,height=512,width=512)
        self.criterion = loss.FocalLoss()
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr, weight_decay=1e-6)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        save_path = main_config['save_path']
        os.makedirs(save_path,exist_ok=True)
        f = open(f'{save_path}/log.txt','w')
        f.close()

        transform1 = A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
            # A.Resize(512,512),
            # A.GaussNoise(var_limit=(50,100)),
            # A.MotionBlur(blur_limit=5),
            A.Blur(5),
            A.OneOf([
                A.ShiftScaleRotate(rotate_limit=(-45,45),p=1),
                # A.ElasticTransform(sigma=30,alpha_affine=30,p=1),
                ],p=0.5),
            ToTensorV2()
        ], additional_targets={'mask2': 'mask', 'mask3': 'mask'})
        transform2 = A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
            # A.Resize(512,512),
            A.GaussNoise(var_limit=(50,100)),
            A.MotionBlur(blur_limit=5),
            A.Blur(5),
            A.OneOf([
                A.ShiftScaleRotate(rotate_limit=(-45,45),p=1),
                A.ElasticTransform(sigma=30,alpha_affine=30,p=1),
                ],p=0.5),
            ToTensorV2()
        ], additional_targets={'mask2': 'mask', 'mask3': 'mask'})
        val_transform = A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
            # A.Resize(512,512),
            ToTensorV2()
        ], additional_targets={'mask2': 'mask', 'mask3': 'mask'})

        train_transform_list = [transform1,transform2,transform1]
        valid_transform_list = [val_transform,val_transform]

        train_dataset_list = [dataset.WifiDataset_segmentation(ann_path,ocr_url,image_path,transform=transform) \
            for ann_path,image_path,transform in \
                zip(main_config['train_json_path_list'],main_config['train_image_path_list'],train_transform_list)]
        val_dataset_list = [dataset.WifiDataset_segmentation(ann_path,ocr_url,image_path,transform=transform) \
            for ann_path,image_path,transform in \
                zip(main_config['val_json_path_list'],main_config['val_image_path_list'],valid_transform_list)]

        train_dataset = dataset.Concat_Dataset(train_dataset_list)
        val_dataset = dataset.Concat_Dataset(val_dataset_list)
        
        # collate_fn needs for batch
        def collate_fn(batch):
            return tuple(zip(*batch))
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=collate_fn)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=collate_fn)


    def train(self):
        print(f'Start training..')
        n_class = 11
        best_loss = 9999999
        
        for epoch in range(self.epochs):
            self.model.train()
            torch.save(self.model,f'{self.save_path}/model.pt')
            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _,_) in enumerate(self.train_dataloader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)
                
                # device 할당
                self.model = self.model.to(self.device)
                
                # inference
                outputs = self.model(images)
                
                # loss 계산 (cross entropy loss)
                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Step [{step+1}/{len(self.train_dataloader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % self.eval_interval == 0 and self.validate:
                avrg_loss = self.validation(epoch + 1, self.model, self.val_dataloader, self.criterion, self.device)
                if avrg_loss < best_loss:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {self.save_path}")
                    best_loss = avrg_loss
                torch.save(self.model.state_dict(),f'{self.save_path}/{epoch+1}.pt')

        torch.save(self.model.state_dict(),f'{self.save_path}/last.pt')


    def validation(self,epoch, model, data_loader, criterion, device):
        print(f'Start validation #{epoch}')
        model.eval()
        
        with torch.no_grad():
            n_class = 11
            total_loss = 0
            cnt = 0
            
            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _,_) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
            

            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            sorted_df = pd.DataFrame({'Categories':['background','ID','PW']})
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]

            
            avrg_loss = total_loss / cnt

            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            save_path = main_config['save_path']
            f = open(f'{save_path}/log.txt','a')
            f.write(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}\n')
            f.close()
            print(f'IoU by class : {IoU_by_class}')
            
        return avrg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, help = 'path of train configuration yaml file')
    args = parser.parse_args()

    with open(args.config) as f:
        main_config = yaml.load(f, Loader = yaml.FullLoader)
    
    Main=main(main_config)
    Main.train()