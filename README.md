# [level2-cv-04] Semantic Segmentation

- Project Period 2023/06/05 ~ 2023/06/22
- [Project Wrap-Up Report](https://docs.google.com/document/d/16whBnd3kEIh85_9x-EVk_o5Str8YRGSKWBGfPgsSdV4/edit?usp=sharing)
  
## **✏️** Project Overview

![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-04/assets/76798969/33010c45-f301-43b2-8ec3-b39ef58f1854)
Bone Segmentation은 인공지능에서 중요한 응용 분야 중 하나로, 다양한 목적으로 도움을 줄 수 있습니다. 이렇게 만들어진 우수한 성능 모델은 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용될 수 있을 것으로 기대됩니다.🌎

- **Input :** hand bone x-ray 객체가 담긴 이미지가 모델의 input으로 사용됩니다.
segmentation annotation은 json file로 제공됩니다.
- **Output :** 모델은 각 픽셀 좌표에 따른 class를 출력하고 이를 rle로 변환하여 리턴합니다.
- **평가지표**: mean Dice
- **프로젝트 주제**: hand bone x-ray 이미지를 29개 항목으로 픽셀에 따른 class 검출
- **프로젝트 구현 내용, 컨셉, 교육 내용과의 관련성**
    - 29가지 범주를 기준으로 객체의 위치를 탐색 및 각 객체를 분류
- **활용 장비 및 재료(개발 환경, 협업 tool 등)**
    - 팀 구성: 4인 1팀
    - 컴퓨팅 환경: 인당 V100 GPU 서버를 VS code와 SSH로 연결하여 사용
    - 협업 툴: notion, git, slack, jira
    - 실험관리: wandb

## 🙌 Members

| 강동화 | 박준서 | 서지희 | 한나영 |
| :---: | :---: | :---: | :---: |
| <img src = "https://user-images.githubusercontent.com/98503567/235584352-e7b0568f-3699-4b6e-869f-cc675631d74c.png" width="120" height="120"> | <img src = "https://user-images.githubusercontent.com/89245460/234033594-cb90a3c0-f0dc-4218-9e11-2abc8db2be67.png" width="120" height="120"> |<img src = "https://user-images.githubusercontent.com/76798969/234210787-18a54ddb-ae13-4554-960e-6bd45d7905fb.png" width="120" height="120"> |<img src = "https://user-images.githubusercontent.com/76798969/233944944-7ff16045-a005-4e4e-bf59-632766194d7f.png" width="120" height="120" />|
| [@oktaylor](https://github.com/oktaylor) | [@Pjunn](https://github.com/Pjunn) | [@muyaaho](https://github.com/muyaaho) | [@Bandi120424](https://github.com/Bandi120424) |



## **🌏** Contributions



| 팀원명 | 학습 모델 | 추가 작업 |
| :---: | :---: | --- |
| 강동화 | FCN, DeepLabV3, DeepLabV3+, UNet++ | EDA, 모델 리서치, pytorch-lightning 실험 환경 세팅, Data Cleansing, Augmentation 실험 및 시각화 구현, RabbitMQ를 사용한 실험 자동화 |
| 박준서 | FCN, DeepLabv3+, HRNet-OCR, UPerNet+ConvNeXt | EDA, 모델 리서치, pytorch-lightning 실험 환경 세팅, mmsegmentation 실험 세팅, Augmentation 리서치 및 실험, 시각화  구현, 모델 앙상블 |
| 서지희 | FCN, UNet, UNet++ | EDA, 모델 리서치, pytorch-lightning 실험 환경 세팅, Augmentation 리서치 및 실험, 시각화 구현, 모델 앙상블 구현 |
| 한나영 | UNet++, FCN | EDA, Jira 세팅, 모델 리서치, mmsegmengtation, smp 및 pytorch-lightning 실험 환경 세팅, SWA ,모델 앙상블 구현 |

![timeline 지금 doc에서 바로 저장이 안되더라구요.. 되면 바로 수정하겠습니다]()

## **❓** Dataset & EDA


- 전체 이미지 개수 : 1100장 (학습 데이터: 800장, 평가 데이터: 300장
- 29 class : finger-1, finger-2, finger-3, finger-4, finger-5, finger-6, finger-7, finger-8, finger-9, finger-10, finger-11, finger-12, finger-13, finger-14, finger-15, finger-16, finger-17, finger-18, finger-19, Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulna
- 이미지 크기 : (2048, 2048)
- 주요 문제점
  |특징|이미지|
  |:---:|:---:|
  |Multi-label classification: 다수의 클래스로 분류해야하는 pixel 존재|![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-04/assets/76798969/69ee56b6-6707-431a-9af1-151477df999b)|
  |Segmentation: 경계가 모호한 사진 존재|![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-04/assets/76798969/8b1f861c-ccc7-4ba7-af34-b254761eed40)|
  |장신구 착용|![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-04/assets/76798969/3b3ebc46-08a7-4b7a-8488-afd9b3a6bfc7)|

- **모델 선정 및 분석**
    - Architecture 
      - **UNet:** Encoder-decoder 기반 모델로 저차원 특징과 고차원 특징 추출
      - **UNet++:** Re-designed skip pathway를 설계함으로써 encoder와 decoder 사이에 semantic gap을 줄여 더 쉽고 빠르게 학습
      - **DeepLabV3:** Atrous Spatial Pyramid Pooling(ASPP)을 통해 Multi-scale contextual feature를 학습
      - **DeepLabV3+:** Decoder에서 backbone의 low-level feature와 ASPP 모듈 출력을 모두 사용하여 단순한 Up-sampling 연산을 개선 
      - **FCN:** semantic segmentation을 위해 고안된 CNN 기반 모델
      - **OCRNet:** 문맥 정보를 고려한 semantic segmentation 모델
      - **UPerNet:** 다양한 visual task 해결을 위해 고안된 FPN 기반 모델
    - HRNet Backbone: High Resolution과 병렬로 Low Resolution을 적용해 전체 stage에서 높은 해상도를 유지합니다. 

## **:scroll: 프로젝트 수행 결과**



![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-04/assets/76798969/2fa1b91a-b3c8-4c8f-a0dd-6a7a4254eb3d)

