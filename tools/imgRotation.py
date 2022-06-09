import requests
import math
import albumentations

from torch import long

def slope(x, y):
    sl = math.sqrt(x**2 + y**2)
    return sl

def get_ann(img_path,api_url) -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": open(img_path  , "rb")}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()

ann_dict = get_ann('../Data set/real data/receipt/receipt040.jpg',
                   "http://118.222.179.32:30000/ocr/")
annos = ann_dict['ocr']['word']

horizontal_list = []

for idx, anno in enumerate(annos):
    xlen = anno['points'][1][0] - anno['points'][0][0]
    # ylen = anno['points'][0][1] - anno['points'][1][1] # 음수일 수도 있음
    horizontal_list.append((xlen, idx))

longest = max(horizontal_list)[1]
print(annos[longest])

thetaplus = True
xlen = annos[longest]['points'][1][0] - annos[longest]['points'][0][0]
ylen = annos[longest]['points'][0][1] - annos[longest]['points'][1][1]

if ylen < 0 :
    thetaplus = False
    ylen = abs(ylen)

costheta = max(horizontal_list)[0] / slope(xlen, ylen)
theta = math.acos(costheta)
degree = round(theta * 57.3,2)
print(degree)



# texts = []
# for ind, anno in enumerate(annos):
#   texts.append((ind, anno['points'][0], anno['text'])) 




# texts_ = sorted(texts, key = lambda x: (x[1][1], x[1][0]))

# new_texts = []
# tmp = texts_[0][1][1]
# phase = []
# new_phase = []
# for text in texts_:
#   if text[1][1] - tmp <= 5: 
#     phase.append(text)
#   else:
#     phase.sort(key = lambda x: x[1][0])
#     new_phase.append([t[2] for t in phase])
#     new_texts.append(new_phase)

#     phase = []
#     tmp = text[1][1]
#     phase.append(text)

# phase.sort(key = lambda x: x[1][0])
# new_phase.append([t[2] for t in phase])
# new_texts.append(new_phase)