import requests

def get_ann(img_path,api_url) -> dict:
    headers = {"secret": "Boostcamp0000"}
    file_dict = {"file": open(img_path  , "rb")}
    response = requests.post(api_url, headers=headers, files=file_dict)
    return response.json()

ann_dict = get_ann('../Data set/real data/receipt/receipt211.jpg',"http://118.222.179.32:30000/ocr/")
annos = ann_dict['ocr']['word']

texts = []
for ind, anno in enumerate(annos):
  texts.append((ind, anno['points'][0], anno['text'])) 

texts_ = sorted(texts, key = lambda x: (x[1][1], x[1][0]))

new_texts = []
tmp = texts_[0][1][1]
phase = []
new_phase = []
for text in texts_:
  if text[1][1] - tmp <= 5: 
    phase.append(text)
  else:
    phase.sort(key = lambda x: x[1][0])
    new_phase.append([t[2] for t in phase])
    new_texts.append(new_phase)

    phase = []
    tmp = text[1][1]
    phase.append(text)

phase.sort(key = lambda x: x[1][0])
new_phase.append([t[2] for t in phase])
new_texts.append(new_phase)

print(new_texts)