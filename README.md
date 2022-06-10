<h1 align="center">Welcome to MODU <img src="https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/wave.gif" width="48px"></h1>
<p>
</p>
<center>
    <img src="./README.assets/logo.png" alt="Wifinder" style="zoom:76%;" align="center"/>
</center>





> Wi-FiNDER (Upstage 기업 연계 프로젝트)

### 🏠 [Github](https://github.com/boostcampaitech3/final-project-level3-cv-04/) :clapper:[Demo 시연영상]() :microphone:[프로젝트 소개 영상]() 📝[Presentation]()

<br>



## ✨ Description

```sh
Wi-FiNDER는 WiFi Image가 들어왔을 때 ID와 PW 값을 추출하여 
사용자가 보다 빠르고 편리하게 WiFi 연결을 할 수 있게 해주는 서비스 입니다.
```





## :pushpin: Project Goal

```sh
카페나 식당과 같은 공공장소에서 제공해주는 WiFi 정보를 좀 더 빠르게 
접근하고, 직
```





## ⏱ Development Schedule

<center>
    <img src="./README.assets/schedule.png" alt="WifiNDER"  align="center"/>
</center>






## 📃 Pipeline

<center>
    <img src="./README.assets/pipeline.png" alt="WifiNDER"  align="center"/>
</center>





## 🗂 Dataset

#### 1. Data Scraping

<center>
    <img src="./README.assets/crawling.png" alt="WifiNDER"/>
</center>

다양한 플랫폼에서 WiFi 정보를 담은 이미지 데이터 수집 후 cleansing 실시

<br>

#### 2. WiFi template

<center>
    <img src="./README.assets/template1.png" alt="WifiNDER"/>
</center>

WiFi template 이미지에 랜덤한 문자열을 ID, PW의 위치에 넣어 합성 이미지 생성

<br>

#### 3. Unity

<center>
    <img src="./README.assets/unity1.png" alt="WifiNDER"/>
</center>

Unity를 이용해 가상 현실로 구현한 카페 Scene에 WiFi 포스터를 다양한 위치에 두고 여러 구도에서 이미지 생성

<br>

#### 4. Data Annotation

<center>
    <img src="./README.assets/annotation.png" alt="WifiNDER"/>
</center>

CVAT Annotation Tool을 활용해 `WiFi Logo`, `WiFi poster`, `ID`, `PW` 에 대한 annotation 진행

<br>






## 🖥Model

### YOLOV5I6 (for Detection)

#### 1. train data

<center style: "">
    <img src="./README.assets/annotation1.png" alt="WifiNDER"  align="center"/>
</center>

WiFi Logo와 WiFi Poster를 annotation한 데이터로 학습

#### 2. poster detection

<center>
    <img src="./README.assets/yolo_output.png" alt="WifiNDER"  align="center"/>
</center>

WiFi Logo가 포함되어 있는 Poster를 detect하고, 해당 영역을 crop하여 사용

<br>



### UNet++ (for Segmentation)

#### 1. train data

<center>
    <img src="./README.assets/annotation2.png" alt="WifiNDER"  align="center"/>
</center>

WiFi ID와 WiFi PW를 annotation한 데이터로 학습

#### 2. 3-channel input

<center>
    <img src="./README.assets/3channel.png" alt="WifiNDER"  align="center"/>
</center>

ch1: gray scale이 적용된 image

ch2: 모든 text 위치의 masking image

ch3: key 값을 가진 text 위치의 masking image



#### 3. Augmentation

- Real Data

<center>
    <img src="./README.assets/scrap_aug.png" alt="WifiNDER"  align="center"/>
</center>

Blur, ShiftScaleRotate 를 통해 이미지 데이터 증강

- Template Data

<center>
    <img src="./README.assets/template_aug.png" alt="WifiNDER"  align="center"/>
</center>

Real Data에 비해 너무 깨끗한 정면 이미지이므로 MotionBlur, ElasticTransform 를 통해 noise 생성

#### 4. input & output & matching

<center>
    <img src="./README.assets/seg_matching.png" alt="WifiNDER"  align="center"/>
</center>

input, output과 output을 ocr bbox와 matching 한 결과

 `id`, `pw`를 key값으로 지니고 있으며, 각각 text 값과 bbox 위치정보를 담아 post processing에 활용

<br>





## 📥 Pre-Processing

### 1. Rotation

<center>
    <img src="./README.assets/4.png" alt="WifiNDER"/>
</center>

그림에 대한 설명

<br>

### 2. Image Padding

<center>
    <img src="./README.assets/5.png" alt="WifiNDER"/>
</center>

그림에 대한 설명

<br>





## 📤 Post-Processing

### 1. Line Alignment (Y-axis)

<center>
    <img src="./README.assets/4.png" alt="WifiNDER"/>
</center>

그림에 대한 설명

<br>

### 2. Word Merge (X-axis)

<center>
    <img src="./README.assets/5.png" alt="WifiNDER"/>
</center>

그림에 대한 설명

<br>

### 3. Final postprocessing

<center>
    <img src="./README.assets/6.png" alt="WifiNDER"/>
</center>

그림에 대한 설명

<br>





## 🖨 Performance

ㅁㄴㅇㄹ





## :wrench: Tech Stack

### Tech Stack

<center>
    <img src="./README.assets/stack.png" alt="WifiNDER"/>
</center>



### System Architecture

<center>
    <img src="./README.assets/arch.png" alt="WifiNDER"/>
</center>



<br>





## :pencil2: ERD

<center>
    <img src="./README.assets/erd.png" alt="WifiNDER"/>
</center>

<br>





## :runner: Steps to run Demo

```bash
$ npm install requirements.txt
$ cd code
$ steamlit run streamlit.py
```

<br>





## 🤼‍♂️Author

🐯**[Roh Hyunsuk](https://github.com/titiman1013)**

🐶 **[Shin Hyeonghwan](https://github.com/vhehduatks)**

🐺 **[Oh Wonju](https://github.com/PancakeCookie)**

🐱 **[Lee Joonhyun](https://github.com/JoonHyun814)**

🦁 **[Lee Hyunsuk](https://github.com/p0tpourri)**

<hr>





## :trophy: Awards

- 





## 📝 License

Copyright © 2022  Sauron's eyes  <br>