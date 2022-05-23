import argparse
from PIL import Image, ImageFont, ImageDraw 
import json
import string
import random
import os
from tqdm import tqdm
import requests
import numpy as np


def random_text():
    string_pool = string.ascii_letters + string.digits + string.punctuation
    string_pool2 = string.ascii_letters + string.digits
    cnt = 0
    length = random.randint(8, 20)
    result = ''
    for _ in range(length):
        if cnt < 3:
            rand_char = random.choice(string_pool)
        else: 
            rand_char = random.choice(string_pool2)
        if rand_char in string.punctuation:
            cnt += 1
        result += rand_char

    return result

def get_ann(img_path,api_url="http://118.222.179.32:30000/ocr/") -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": open(img_path  , "rb")}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()

def get_keys(ann):
    id_list = ['ID', 'ID:', '아이디', '아이디:', 'NETWORK']
    pw_list = ['PW', 'PW:', '비밀번호', '비밀번호:', 'PASSCODE:', 'password', 'password:', 'PASSWORD', 'PASSWORD:']
    words = ann['ocr']['word']
    position_id = [0,0,0,0]
    position_pw = [0,0,0,0]
    for word in words:
        if word['text'] in id_list:
            position_id = word['points']
        if word['text'] in pw_list:
            position_pw = word['points']

    return position_id, position_pw

def get_text_color(template_img, pos_id_key, gen_id_bbox):
    pix = template_img.load()
    x1, y1 = pos_id_key[0]
    x2, y2 = pos_id_key[2]
    x1_, y1_ = gen_id_bbox[0], gen_id_bbox[1]
    x2_, y2_ = gen_id_bbox[2], gen_id_bbox[3]

    color_list = []
    for i in range(x1, x2+1): # ID bbox의 pixel값들 저장
        for j in range(y1, y2+1):
            if not isinstance(pix[i,j], int) and len(pix[i,j]) == 4:
                pix_c = (pix[i,j][0:3])
            else:
                pix_c = pix[i,j]
            color_list.append(pix_c)

    for i in range(x1_, x2_+1): # 생성할 위치의 bbox의 pixel값들 저장(배경 관련 rgb값을 더해주기 위한 목적)
        for j in range(y1_, y2_+1):
            if not isinstance(pix[i,j], int) and len(pix[i,j]) == 4:
                pix_c = (pix[i,j][0:3])
            else:
                pix_c = pix[i,j]
            color_list.append(pix_c)
    color_list_ = np.array(color_list)
    mean_color = np.mean(color_list_, 0) # 배경색과 가까운 rgb 평균값을 얻을 수 있음.
    d_list = []
    for color in color_list_:
        D = np.sqrt(np.sum((color - mean_color) ** 2)) # 평균과 각 색과의 유클리드 거리 계산
        d_list.append(D)
    ind = d_list.index(max(d_list)) # 평균 색에서 가장 거리가 먼 색을 text color라 가정
    return color_list[ind]

def wifi_image_generation():

    img_path = arg.img_path
    font_path = arg.font_path 
    num_img = arg.num_img
    text_id = arg.text_id 
    text_pw = arg.text_pw
    text_size = 10
    text_color = (0, 0, 0)
    
    # Load image & annotation
    image_id = 0
    box_id = 1
    coco_img_list = []
    coco_ann_list = []

    img_list = os.listdir(img_path)
    for img_name in tqdm(img_list):

        filename = os.path.join(img_path,img_name)
        template_ann = get_ann(filename)
        pos_id_key, pos_pw_key = get_keys(template_ann)
        pos_x_gen_id = pos_id_key[1][0]
        pos_x_gen_pw = pos_pw_key[1][0]
        pos_y_gen_id = min(pos_id_key[0][1], pos_id_key[1][1])
        pos_y_gen_pw = min(pos_pw_key[0][1], pos_pw_key[1][1])

        for n in range(num_img):
            image_id += 1
            template_img = Image.open(filename)
            img_width, img_height = template_img.size

            # 생성할 문자열의 좌상단 좌표, key와의 margin을 random값으로 지정(10~20 pixels)
            pos_id_value = [pos_x_gen_id + random.randint(10, 20), pos_y_gen_id] 
            pos_pw_value = [pos_x_gen_id + random.randint(10, 20), pos_y_gen_pw] 
    
            # Font Selection
            font_list = os.listdir(font_path)
            fontfile = font_path + '/' + random.choice(font_list)
            text_size = max(pos_id_key[3][1] - pos_id_key[0][1], pos_id_key[2][1] - pos_id_key[1][1])
            font = ImageFont.truetype(fontfile, text_size)

            # Render the Text
            image_editable = ImageDraw.Draw(template_img) # Convert the image to an editable format
            if arg.text_id is None:
                text_id = random_text()
            if arg.text_pw is None:
                text_pw = random_text()

            # Starting Coordinates:(0,0) in the upper left corner, Text, Text color: RGB, Font style
            gen_id_bbox = image_editable.textbbox(pos_id_value, text_id, font=font) # (x0,y0,x2,y2)
            gen_pw_bbox = image_editable.textbbox(pos_pw_value, text_pw, font=font) # (x0,y0,x2,y2)
            text_color = get_text_color(template_img, pos_id_key, gen_id_bbox)
            image_editable.text(gen_id_bbox, text_id, text_color, font=font) # ID
            image_editable.text(gen_pw_bbox, text_pw, text_color, font=font) # PW

            # Export the result
            export_dir = './gen_imgs'
            if os.path.isdir(export_dir) == False:
                os.mkdir(export_dir)
            export_filename = img_name[0:img_name.rfind('.')] + '_' + str(n).zfill(2) +'.jpg'
            template_img.save('/'.join([export_dir, export_filename]))

            # Make COCO format annotation
            coco_img_list.append(coco_img(image_id, img_width, img_height, export_filename))
            box_id, ann_info = coco_ann(gen_id_bbox, gen_pw_bbox, box_id, image_id)
            coco_ann_list.extend(ann_info)

    print(f'{len(img_list) * num_img} wifi images generated')

    coco_dataset = coco_template()
    coco_dataset['images'] = coco_img_list
    coco_dataset['annotations'] = coco_ann_list
    anno_dir = './gen_imgs/annotations'
    if os.path.isdir(anno_dir) == False:
        os.mkdir(anno_dir)
    with open('./gen_imgs/annotations/anno.json','w') as f:
        json.dump(coco_dataset, f, indent=4)
    print('COCO format annotations file generated')

def coco_template():
    coco_dataset = {
    "licenses": [
        {
        "name": "",
        "id": 0,
        "url": ""
        }
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [
        {
        "id": 1,
        "name": "ID",
        "supercategory": ""
        },
        {
        "id": 2,
        "name": "PW",
        "supercategory": ""
        }
    ]
    }
    return coco_dataset

def convert_coco_bbox(bbox):
    width = bbox[2] - bbox[0] 
    height = bbox[3] - bbox[1]
    
    return [bbox[0], bbox[1], width, height], width * height

def coco_img(image_id, width, height, filename): 
    img_info = {
      "id": image_id,
      "width": width,
      "height": height,
      "file_name": filename,
      "license": 0,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": 0
    }
    return img_info

def coco_ann(id_bbox, pw_bbox, box_id, image_id):

    id_bbox, id_area = convert_coco_bbox(id_bbox)
    pw_bbox, pw_area = convert_coco_bbox(pw_bbox)

    ann_info = [{
      "id": box_id,
      "image_id": image_id,
      "category_id": 1, # ID
      "segmentation": [
          [
          id_bbox[0], id_bbox[1],
          id_bbox[0] + id_bbox[2], id_bbox[1],
          id_bbox[0] + id_bbox[2], id_bbox[1] + id_bbox[3],
          id_bbox[0], id_bbox[1] + id_bbox[3]
          ]
      ],
      "area": id_area,
      "bbox": id_bbox,
      "iscrowd": 0,
      "attributes": {
        "occluded": False
      }
    },
    {
      "id": box_id+1,
      "image_id": image_id,
      "category_id": 2, # PW
      "segmentation": [
          [
          pw_bbox[0], pw_bbox[1],
          pw_bbox[0] + pw_bbox[2], pw_bbox[1],
          pw_bbox[0] + pw_bbox[2], pw_bbox[1] + pw_bbox[3],
          pw_bbox[0], pw_bbox[1] + pw_bbox[3]
          ]
      ],
      "area": pw_area,
      "bbox": pw_bbox,
      "iscrowd": 0,
      "attributes": {
        "occluded": False
      }
    }]
    return box_id + 2, ann_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-img','--img_path',default='./templates')
    parser.add_argument('-font','--font_path',default='./fonts')
    parser.add_argument('-num','--num_img',default=3)
    parser.add_argument('-id','--text_id',default=None)
    parser.add_argument('-pw','--text_pw',default=None)
    arg = parser.parse_args()

    wifi_image_generation()