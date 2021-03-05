Deep Learning Project

Tracking people in CCTV using YOLO
==================================

I. Introduction
----------------
현대사회에서 CCTV는 보안 및 치안 등의 목적으로 곳곳에 설치되어 있다. 사람들의 편리와 안전을 지켜주기 위해 하루에도 엄청난 양의 데이터를 쏟아내고 있지만, 미아가 생기고 범죄 사건이 발생한다. 본 프로젝트에서는, YOLO를 활용한 Object Tracking으로 미아 혹은 수상한 사람을 CCTV에서 검출하여 다른 각도의 CCTV에서 발견한 동일 인물을 검출하도록 노력하였다. YOLO의 여러 버전들을 비교 분석하며, 목표 인물이 영상에 검출 되었다면 실제 사람이 해당 영상을 통해 확인하고 추적하는 컨셉이기 때문에 끊기지 않는 추적보다는 목표 인물 검출에 포커스를 두었다.
   
#### 데이터셋 소개: 실내 50개, 실외 50개 총 100개의 폴더로 구성
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108450701-fa0a8e00-72a8-11eb-82f0-0d82da5d6924.png" width="780" height="280"></p>

* 폴더 구성
  * video : 추적대상이 찍힌 영상 파일 
  * json : json형식의 bounding box 좌표
  * frames : 추적대상의 이미지 파일
 
II. Result
-----------
1. 프로젝트 결과  
- indoor 1 : 17개의 frame 학습 후 test (핑크 자켓 여성)
- indoor 2 : 9개의 frame 학습 후 test (빨간 줄무늬티 남자아이)
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108315152-5a90c100-71fe-11eb-82eb-712fbe3c8ca2.gif" width="390" height="230"/> <img src="https://user-images.githubusercontent.com/72811950/108521683-52747680-730f-11eb-9878-aa6a2bf74b04.gif" width="390" height="230"/></p>

- outdoor 1 : 16개의 frame 학습 후 test (중절모 남성)
- outdoor 2 : 18개의 frame 학습 후 test (분홍티 여자아이)
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108521716-5d2f0b80-730f-11eb-8975-89f91028134a.gif" width="390" height="230"/> <img src="https://user-images.githubusercontent.com/72811950/108314491-5617d880-71fd-11eb-925d-a49820d311f0.gif" width="390" height="230"/></p>

- 프로젝트 결과 전체 영상  
  indoor 1 : <https://youtu.be/EPoV2Pz7U2Y>  
  indoor 2 : <https://youtu.be/q-gwFKXi6mQ>  
  outdoor 1 : <https://youtu.be/vFO4J6Q2Ts8>  
  outdoor 2 : <https://youtu.be/Uwu12zHNlns>  
  
2. Comparison of YOLO performance
- 아래는 버전별로 같은 이미지를 학습한 후 테스트한 결과이다.  
  같은 이미지를 학습했지만 버전별로 튜닝한 부분이 약간씩 다르고 YOLOv5는 YOLOv3, YOLOv4와 다르게 darknet기반이 아니어서 사용방법이 달라 같은 환경에서 학습되었다고 할 수는 없다. 하지만 우리 데이터에 맞게 튜닝 후 사용했을 때의 성능이므로 참고할 수 있을 것이다.

- 비교 영상
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108838061-7ccb7a00-7616-11eb-860c-384335ef01d4.gif" width="260" height="150"/> <img src="https://user-images.githubusercontent.com/72811950/108838063-7dfca700-7616-11eb-870e-de6615398aaf.gif" width="260" height="150"/> <img src="https://user-images.githubusercontent.com/72811950/108838065-7e953d80-7616-11eb-84b7-41d5743b9b84.gif" width="260" height="150"/></p> 
<h5 align="center">yolov3　　　　　　　　　　　　　　　　yolov4　　　　　　　　　　　　　　　　yolov5</h5>

- 전체 영상  
  YOLOv3 : <https://youtu.be/z-6r845s_a0>  
  YOLOv4 : <https://youtu.be/gO6YRpdwm4k>  
  YOLOv5 : <https://youtu.be/kEYcr76nE00>
  
- 그래프
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108842495-b3a48e80-761c-11eb-809d-e13b0ade36cd.png" width="650" height="380"></p>
  
III. Process
-------------
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108320197-d17d8800-7205-11eb-9265-297ef37e5a0a.png" width="780" height="180"></p>

1. Preprocessing
   * Image Augmentation
      - [Image augmentation code](기중 이미지 증강 시키는 코드 커밋하고 여기에 코드 url 넣어주세여)
      <p align="center"><img src="https://user-images.githubusercontent.com/72811950/108452462-e3196b00-72ab-11eb-9472-0caae061ef4a.jpg" width="780" height="400"></p>

   * json -> txt
      - [Format conversion code](https://github.com/yeji0701/DeepLearning_Project/blob/main/code/jc/00_label_json_to_txt.ipynb)
      ![image](https://user-images.githubusercontent.com/28764376/108456228-37741900-72b3-11eb-87ad-d6dab055b416.png)

2. Training
   * Changing Resolution Size : [Code]()
   ```
   <yolo.cfg>

   # 정확도 향상을 위해 픽셀 해상도를 크게 함

   batch=64
   subdivisions=32
   width=608  <-- 변경
   height=608  <-- 변경
   ```
   * Optimizing Anchor Box : [Code]()
   ```
   from utils.autoanchor import *
   _ = kmean_anchors(path='./data.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)
   ```
3. Test
   * Adjusting Confidence Threshold : [Code]()
   ```
   # -threshold {} <-- 조정하여 되도록 target만 detection하도록 함
   
   ./darknet detector demo custom_data/detector.data custom_data/cfg/yolov3-custom-test.cfg 
   backup/yolov3-custom_best.weights ./test.mp4 -thresh 0.6 -out_filename out.avi -dont_show
   ```

IV. Customed performance evaluation
-----------------------------------
- mAP(Mean Average Precision)는 Object Detection의 평가지표로 많이 사용된다. 본 프로젝트에서도 이런 정량적 평가를 통해 test결과를 평가하려 했으나 한계가 있었다. 목표 target이 아니라 타물체를 감지한 경우에도 계산되어 mAP가 높게 나오는 경우 때문이었다. 따라서 우리는 자체 수기 산출 방식을 고안하였고 test영상에서 초당 3개의 frame을 sampling하여 수기로 Accuracy, Precision, Recall, F1-score을 산출 하였다.

   * [Detailed scoreboard](https://github.com/yeji0701/DeepLearning_Project/blob/main/scoreboard.xlsx)
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/109183423-45ed9380-77d1-11eb-9760-daf18d7860a9.png" width="700" height="400"></p>


마치며
------
- 배운점
1. 다양한 이미지 증강 기법  
2. Object detection 알고리즘의 발전  
3. YOLO 모델의 원리 학습  
4. IoU, mAP 등 Object detection 평가 지표에 대한 이해

- 개선할 점
1. yolo 소스 코드 분석을 통한 데이터에 최적화 된 튜닝
2. 보다 Target에 특화된 Custom Training을 통한 모델 개선
3. 모든 시도는 소중하니, 결과를 기록하는 습관 개선

Built with
-----------
- 김미정
   * 이미지 증강 
   * YOLOv3-tiny, YOLOv3, YOLOv4-tiny, YOLOv4, YOLOv5를 이용한 object tracking
   * YOLOv3, YOLOv4, YOLOv5 성능비교
   * YOLOv4+DeepSort 이용한 object tracking
   * Github: https://github.com/LeilaYK
- 김예지
   * 이미지 증강 및 mixup, YOLOv3-tiny와 YOLOv3를 이용한 object tracking
   * Github: https://github.com/yeji0701
- 이기중
   * 이미지 증강 및 yolov3 yolov4를 이용한 object detection 테스트
   * Github: https://github.com/GIGI123422
- 최재철
   * yolov3-tiny
   * yolov5 config setting, scoreboard sheet
   * Github: https://github.com/kkobooc

Acknowledgements
-----------------
- [Darknet](https://github.com/pjreddie/darknet)
- [Alexey](https://github.com/AlexeyAB)
- [YOLOv4](https://github.com/kiyoshiiriemon/yolov4_darknet)
- [YOLOv4 + DeepSORT](https://github.com/theAIGuysCode/yolov4-deepsort)
- [YOLOv5](https://github.com/ultralytics/yolov5)
