import io
from tkinter import image_names
import requests
import streamlit as st
from itertools import chain
import copy

from PIL import ImageOps, Image, ImageDraw, ImageFont

ocr_url = "http://118.222.179.32:30000/ocr/"
font_path='/opt/ml/final-project-level3-cv-04/tools/fonts/NanumSquareRoundB.ttf'

def get_ann(img,api_url:str) -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": img}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()


def draw_polygon(img,box_args_list,area_list):
    """이미지에 폴리곤을 그린다. illegibility의 여부에 따라 라인 색상이 다르다."""

    img_draw = ImageDraw.Draw(img,'RGBA')
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


key_words=['Pass','PASS']



def read_img(image,ocr_url,target_h: int = 1000) -> Image:
    # load image, annotation

    image_bytes = image.getvalue()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert('RGBA')  # 이미지 정보에 따라 이미지를 회전
    area = None
    area_resize= None
    box_args_list=[]
    area_list=[]

    ann_dict = get_ann(image_bytes,ocr_url)

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
        
        for key in key_words:
            if key in tag_text:
                area = copy.deepcopy(poly)
                area[0],area[1],area[2],area[3]=[0,area[0][1]-word_height],[w,area[1][1]-word_height],[w,area[2][1]+word_height*2],[0,area[2][1]+word_height*2]
                area_resize = [list(map(lambda x: x*ratio,temp)) for temp in area]
   
        poly_resize = [list(map(lambda x: x*ratio,temp)) for temp in poly]
        box_args_list.append({"poly_resize":poly_resize, "tag_text":tag_text, "tag_ori":tag_ori})
        if area_resize:area_list.append(area_resize)
        
    draw_polygon(img,box_args_list,area_list)
    return img , ann_dict

def main():
    uploaded_file = st.file_uploader("img",type=['png','jpg','jpeg'])

    if uploaded_file:
        image,ann_dict = read_img(uploaded_file,ocr_url)

        st.write(ann_dict)
        st.image(image, caption='Uploaded Image')



if __name__ == "__main__":
	main()
