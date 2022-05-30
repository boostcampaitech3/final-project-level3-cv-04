import requests

def get_ann(img_path,api_url) -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": open(img_path  , "rb")}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()

ann_dict = get_ann('output.png',"http://118.222.179.32:30000/ocr/")
# print(ann_dict)
annos = ann_dict['ocr']['word']
# print(annos)

texts = []
for ind, anno in enumerate(annos):
  # print((ind, anno['points'][0], anno['text']))
  texts.append((ind, anno['points'][0], anno['text'])) 

texts_ = sorted(texts, key = lambda x: (x[1][1], x[1][0]))  # y 먼저 그다음 x

new_texts = []
tmp = texts_[0][1][1] # 첫번째 글자의 y좌표 
phase = []
new_phase = []
for text in texts_:
  if text[1][1] - tmp <= 12:  #  같은 라인 판별
    phase.append(text)
  else: # tmp값 벌어지면 다음 라인 취급
    phase.sort(key = lambda x: x[1][0]) # x로 정렬
    new_phase.append([t[2] for t in phase])
    new_texts.append(new_phase) # 한 문장으로 저장

    phase = []
    tmp = text[1][1]
    phase.append(text)

phase.sort(key = lambda x: x[1][0])
new_phase.append([t[2] for t in phase])
new_texts.append(new_phase)

print(new_texts)