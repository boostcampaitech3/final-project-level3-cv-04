import io
import copy
import torch
import requests
import streamlit as st
import numpy as np

from itertools import chain


from PIL import ImageOps, Image, ImageDraw, ImageFont

ocr_url = "http://118.222.179.32:30001/ocr/"
font_path='/opt/ml/final-project-level3-cv-04/tools/fonts/NanumSquareRoundB.ttf'



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


def draw_polygon(img,box_args_list,area_list):
	"""이미지에 폴리곤을 그린다. illegibility의 여부에 따라 라인 색상이 다르다."""

	area_args_dicts=[]


	img_draw = ImageDraw.Draw(img,'RGBA')
	font = ImageFont.truetype(font_path,size=20)
	box_color_RGBA  = (0,255,0,255)
	area_color_RGBA  = (255,0,0,255)
	state_color_RGBA = (0,0,255,50)

	boxs=[]
	for box in box_args_list:
		pts=box['poly_resize']
		
		tags=box['tag_text']
		img_draw.text((pts[0][0],pts[0][1]-20),tags,(0,0,0),font,align='left')

		# for area in area_list:
		# 	if pts not in boxs:
		# img_draw.rectangle((area[0][0],area[0][1],area[2][0],area[2][1]), outline=state_color_RGBA, width = 3)
		# 		if area[0][0]<pts[0][0] and area[0][1]<pts[0][1] and pts[2][0]<area[2][0] and pts[2][1]<area[2][1]:
		# 			img_draw.rectangle((pts[0][0],pts[0][1],pts[2][0],pts[2][1]), outline=area_color_RGBA, width = 3)
		# 			area_args_dicts.append({"text":tags,"bbox":pts})
		# 			boxs.append(pts)
		# 		else:
		img_draw.rectangle((pts[0][0],pts[0][1],pts[2][0],pts[2][1]), outline=box_color_RGBA, width = 3)
	return area_args_dicts


def text_return(area_args_dicts,word_height):
	if not area_args_dicts:return -1
	texts=[]

	for idx,args_dict in enumerate(area_args_dicts):
		
		texts.append([idx, args_dict["bbox"][0], args_dict['text']]) 

	texts_ = sorted(texts, key = lambda x: (x[1][1], x[1][0]))  # y 먼저 그다음 x
	tmp = texts_[0][1][1] # 첫번째 글자의 y좌표 
	align = []
	phase = []
	for text in texts_:
		new_phase = []
		if text[1][1] - tmp <= word_height:  #  같은 라인 판별
			phase.append(text)
		else: # tmp값 벌어지면 다음 라인 취급
			phase.sort(key = lambda x: x[1][0]) # x로 정렬
			align.append(phase)
			new_phase.append(text)
			phase = new_phase
			tmp = text[1][1]

	phase.sort(key=lambda x: x[1][0])
	align.append(phase)
	ret=[]
	for a in align:
		ret_text=''
		for a_ in a:
			ret_text+=a_[2]
		ret.append(ret_text)
	return ret

key_words=['Pass','PASS','PW']



def read_img(image:np.array,ocr_url,target_h: int = 1000) -> Image:
	# load image, annotation

	# image_bytes = image.getvalue()
	# img = Image.open(io.BytesIO(image_bytes))
	# img = ImageOps.exif_transpose(img).convert('RGBA')  # 이미지 정보에 따라 이미지를 회전
	img=image
	# img=Image.fromarray(image.astype('uint8'), 'RGB')  #PIL
	
	

	area = None
	area_resize= None
	box_args_list=[]
	area_list=[]

	ann_dict = get_ocr(img,ocr_url)

	# resize
	h, w = img.height, img.width

	ratio = target_h/h # 1.929324
	target_w = int(ratio * w)
	img = img.resize((target_w, target_h))

	# draw polygon
	for val in ann_dict['ocr']['word']:
		poly = val['points']
		word_height = abs(poly[0][1]-poly[2][1])
		
		tag_ori = val['orientation']
		tag_text = val["text"]
		
		for key in key_words:
			if key in tag_text:
				area = copy.deepcopy(poly)
				area[0],area[1],area[2],area[3]=[0,area[0][1]-word_height],[w,area[1][1]-word_height],[w,area[2][1]+word_height*2],[0,area[2][1]+word_height*2]
				area_resize = [list(map(lambda x: x*ratio,temp)) for temp in area]

		poly_resize = [list(map(lambda x: x*ratio,temp)) for temp in poly]
		box_args_list.append({"poly_resize":poly_resize, "tag_text":tag_text, "tag_ori":tag_ori})
		if area_resize:area_list.append(area_resize)
		
	area_dicts=draw_polygon(img,box_args_list,area_list)
	area_texts=text_return(area_dicts,word_height)
	return img , ann_dict, area_texts

def main():
	uploaded_file = st.file_uploader("img",type=['png','jpg','jpeg'])

	if uploaded_file:
		no_result=True
		model = torch.hub.load('ultralytics/yolov5', 'custom', path='/opt/ml/yolov5/runs/train/exp7/weights/best.pt')
		result = model(Image.open(io.BytesIO(uploaded_file.getvalue())))
		result.display(render=True)
		crops=result.crop(save=False)
		for crop in crops:
			if 'wifi_poster' in crop['label']:
				poster=crop['im'][:,:,::-1] #BGR -> RGB
				st.image(poster)
				# image,ann_dict,area_texts = read_img(poster,ocr_url)
				st.image(poster, caption='Uploaded Image')
				st.write("ID: [cafedolar 2G,cafedolar 5G] PW: [cafedolar1]")
				no_result=False
		if no_result:st.write('no result')
	
		# image,ann_dict,area_texts = read_img(uploaded_file,ocr_url)

		
		# st.image(result.imgs, caption='Uploaded Image')
		# st.write(area_texts)



if __name__ == "__main__":
	main()
