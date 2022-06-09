from unicodedata import name
import custom_utils
import dataset
import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st
import requests
import io
import rule_based_method as rule
import json

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
		if abs(text[1][1] - tmp) <= ((text[5][1] - text[1][1])/1.5) :  #  같은 라인 판별 / 글자의 반 이내면 
			phase.append(text)
			tmp = min(text[2][1], text[1][1])
		else: # tmp값 벌어지면 다음 라인 취급
			phase.sort(key = lambda x: x[1][0]) # x로 정렬
			align.append(phase)
			new_phase.append(text)
			phase = new_phase
			tmp = min(text[2][1], text[1][1])

	phase.sort(key = lambda x: x[1][0])
	align.append(phase) # 마지막 줄 추가


	print("--------------------")
	line = []
	word = []
	for i in align:
		tmp = i[0][1][0]
		for n in i:
			if n[1][0] - tmp <= ((n[2][0]-n[1][0])/len(n[3]))/1.5: 
				word.append(n[3])
				tmp = n[2][0]
			else:
				word.append(" ")
				word.append(n[3])
				tmp = n[2][0]
		line.append(word)
		word = []
	
	out = []
	for i in line:
		s = "".join(i)
		print(s)
		out.append(s)

	return out


def get_3chanel_key_masked_image(image,ocr,img_path):
	ocr_coco = custom_utils.ocr_to_coco(ocr,img_path,(image.shape[0],image.shape[1]))
	c2 = custom_utils.coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=False)
	c3 = custom_utils.coco_to_mask(ocr_coco,image.shape,key_list=dataset.key_list,get_each_mask=False)
	_,mask_list = custom_utils.coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=True)
	print(torch.tensor(image).unsqueeze(0).shape)
	return torch.cat((torch.tensor(image).unsqueeze(0),c2,c3),dim=0), mask_list


def pipeline(img,model,device):
	### 1. img_path --> PIL image ###
	# image = Image.open(img_path)
	# image = ImageOps.exif_transpose(image).convert('L')      #image:PIL
	image_3c = Image.fromarray(img.astype('uint8'), 'RGB')
	image_g = ImageOps.exif_transpose(image_3c).convert('L')

	### 2. PIL image rotate ###
	image = custom_utils.img_rotate(image_g)             #image:np.array
	
	### 3. get ocr ###
	ocr = custom_utils.get_ocr(Image.fromarray(image),"http://118.222.179.32:30001/ocr/")

	### 4. get key masked image, each mask list ###
	x, mask_list = get_3chanel_key_masked_image(image,ocr,'None')     # x:torch.tensor
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
	classificated_image,out_list = custom_utils.seg_to_classification(pred,t_ocr_list,device)

	if out_list['id']:
		out_list['id'] = bbox_concat(out_list['id'])
	if out_list['pw']:
		out_list['pw'] = bbox_concat(out_list['pw'])

	fin_out = {}
	fin_out['id'] = [out for out in out_list['id']]
	fin_out['pw'] = [out for out in out_list['pw']]


	return classificated_image,fin_out['id'],fin_out['pw'],Image.fromarray(custom_utils.img_rotate(image_3c).astype('uint8'), 'RGB')

def output_func(poster):
	st.write(uploaded_file.name)
	poster=poster['im'][:,:,::-1] #BGR -> RGB
	ret_img,ret_id,ret_pw,crop_img=pipeline(poster,seg_model,device) # ret_img : Tensor
	ret_img=torchvision.transforms.ToPILImage()(ret_img) 
	output = io.BytesIO()
	image = ret_img
	image.save(output, format="JPEG")
	ocr_img,ann_dict,area_texts = rule.read_img(crop_img,'http://118.222.179.32:30001/ocr/')
	st.image(ocr_img,caption='ocr Image')
	# st.image(image, caption='after pipeline Image') # seg output
	ret_id = ", ".join(ret_id)
	ret_pw = ", ".join(ret_pw)

	id=st.text_input('ID',ret_id)
	pw=st.text_input('PW',ret_pw)
	check = st.checkbox('check string')
	if check:
		if st.button('submit'):
			save_path='./user_data'
			user_dict={'user_anno_id':id,'user_anno_pw':pw}
			file_name=uploaded_file.name.split('.')[0]
			crop_img.save(os.path.join(save_path,uploaded_file.name))
			with open(os.path.join(save_path,f'{file_name}.json'),'w') as f:
				json.dump(user_dict, f)
			qr=custom_utils.wifi_qrcode(id,'true','WPA',pw)
			st.image(qr)

if __name__ == '__main__':
	with st.sidebar:
		uploaded_file = st.file_uploader("img",type=['png','jpg','jpeg'])

	if uploaded_file:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		seg_model = torch.load('/opt/ml/upstage_OCR/code/saved/seg_model/model.pt')
		seg_model.load_state_dict(torch.load('/opt/ml/upstage_OCR/code/saved/seg_model/seg_c1_k2.pt'))
		det_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/opt/ml/upstage_OCR/code/saved/det_model/yolov5s_wifi_det.pt')


		input_img=Image.open(io.BytesIO(uploaded_file.getvalue()))
		result = det_model(input_img)
		result.display(render=False)	
		crops=result.crop(save=False)
		
		posters=[]
		logos=[]
		for crop in crops:
			if int(crop['cls'].item())==2:
				posters.append(crop)
			else:
				logos.append(crop)

		for poster in posters:	
			poster_upx=poster['box'][0].item()
			poster_upy=poster['box'][1].item()
			poster_downx=poster['box'][2].item()
			poster_downy=poster['box'][3].item()

			for logo in logos:
				logo_upx=logo['box'][0].item()
				logo_upy=logo['box'][1].item()
				logo_downx=logo['box'][2].item()
				logo_downy=logo['box'][3].item()

				if poster_upx<logo_upx and poster_upy<logo_upy and logo_downx<poster_downx and logo_downy<poster_downy:
					output_func(poster)

			if not logo:
				output_func(poster)


### TODO : logo 가 없을때, 출력을 표로바꿈, 수정하기 버튼 추가->사용자입력 추가, 



# streamlit run pipeline.py --server.port 30001 --server.fileWatcherType none
