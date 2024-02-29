# StarGAN-FE (StarGAN-Facial Expression)

## 자폐 아동을 위한 감정 생성 AI

![Untitled](https://github.com/seoin0110/ADDI_StarGan/assets/79834222/e2dbd890-4063-4cfe-978a-1343453cb400)

## **I.** Intro

### Objective

자폐 스펙트럼 장애를 가진 아동들은 종종 타인의 감정을 이해하거나 상황에 따른 적절한 감정을 표현하는 데 어려움을 겪는다. 자폐 아동의 감정 인식 및 표현 능력을 향상시키기 위해, 감정을 분석하고 이해하는 데 도움을 주는 감정 생성 AI를 개발한다. 자폐 아동이 다양한 감정을 인식하고, 자신의 감정을 적절히 표현하는 데 필요한 지원을 제공하는 것을 목적으로 한다. AI는 컴퓨터 비전 분야의 생성 AI 기술을 기반으로 한다.

### Overview

StarGAN-TA 모델을 활용한다. AI Hub의 한국인 감정인식을 위한 복합 영상 데이터셋에서 [기쁨, 당황, 분노, 불안, 슬픔] 5가지 라벨을 사용하였고, 각 라벨당 약 70000장의 이미지로 학습하였다.

StarGAN-TA 학습 시에 데이터는 얼굴만 crop하여 512\*512로 resize하는 전처리를 거쳤고(`datapreprocess.py`), 학습 시에는 256\*256으로 다시 resize하여 학습하였다.

StarGAN-TA 추론 시에도 학습 때와 같은 전처리 과정을 거친 후 Generator 모델을 통해 추론 된 결과를 다시 원본 사이즈로 resize 후에 mediapipe의 face_detection을 통해 face mesh를 찾고 안쪽 부분만 생성된 이미지로 대체한다. (`face_detection.py`)

best checkpoint를 저장하는 metric으로 FID score과 감정인식모델([enet_b2_7](https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b2_7/?raw=true)) 결과를 곱하여 사용하였다.

## **II.** Architecture Details

`data_loader.py`

- sample과 train data를 load 한다.
- 밝은 이미지에 대해 생성을 어둡게 하는 경향이 있어 ColorJitter transform을 추가하였다.
- T.ColorJitter(brightness=(0.8, 1.4), contrast=0.2)

`face_detection.py`

- **detection_and_resize_original**(image, size)
  - test 시에 이미지에서 face의 x, y, h, w 좌표를 찾아주어 얼굴 이미지만 crop한 후 사용한다.
- **get_face_mesh**(img)
  - face_mesh를 찾아주고, 생성된 이미지에서 face_mesh에 해당하는 부분만 자연스럽게 원본 이미지를 대체한다.
  - ‘분노’ 감정의 특성상 생성된 이미지의 눈썹이 아래로 이동하여 원본 이미지의 눈썹과 중복하여 표현하는 경우가 있어 face_mesh에서 이마 부분을 얼굴 길이의 0.2배만큼 넓게 잡도록 수정하였다.

`model.py`

- 모델 구조는 [StarGAN-TA](https://github.com/shp216/StarGAN-TA) 깃허브와 [StarGAN](https://arxiv.org/pdf/1711.09020.pdf) 논문을 참고한다.

`solver.py`

- sample_step마다 train batch 이미지를 test하여 sample 이미지로 제공한다.
- 생성된 sample 이미지로 FID score와 `emotion_recognition.py`의 감정인식모델을 활용해 ckpt 저장 여부를 판단하는 metric으로 사용한다.
- 즉, best-G.ckpt와 best-D.ckpt가 갱되는 주기는 sample_step과 같다.

## III. Environment Set-up

### Step1. Download this model

`git clone` 을 사용해 프로젝트를 다운받을 수 있다.

```
git clone https://github.com/seoin0110/StarGAN-FE.git
cd StarGAN-FE
```

미리 학습된 모델 체크포인트는 [링크](https://drive.google.com/drive/folders/1cGWF-CcUR04r6YbZLTP5Q84nNrqCUmyy?usp=drive_link)에서 다운 받아 사용한다.

```
stargan
├── backend.py
├── data_loader.py
├── datapreprocess.py
├── emotion_recognition.py
├── data
│   ├── train
│   │   ├── angry
│   │   ├── fearful
│   │   ├── happy
│   │   ├── sadness
│   │   └── surprised
│   ├── test
│   ├── models
│   │   ├── ex) 600000-D.ckpt
│   │   └── ex) 600000-G.ckpt
│   ├── samples
│   │   └── ex) 600000-images.jpg
│   └── logs
├── emotion_recognition.py
├── face_detection.py
├── facemesh.py
├── main.py
├── model.py
├── README.md
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
├── solver.py
└── streamlit_app.py
```

### Step2. Create anaconda virtual environment

conda를 사용하여 가상환경을 설정한다.

env_name 자리에 가상환경 이름을 다음과 같이 설정할 수 있다. ex) StarGAN, virtual_star, …

이때 파이썬 버전은 3.7로 한다.

```
(base) ~/stargan> conda create -n env_name python=3.7
(base) ~/stargan> conda activate env_name
(env_name) ~/stargan> pip install -r requirements.txt

```

### Step3. train & test

모든 config 변수들은 main.py 실행 시 parser를 통해 받는다.

다음과 같은 변수를 설정할 수 있다.

```json
{
	'c_dim': 5, (label의 개수 -> 생성할 감정의 종류)
	'crop_size': 448,
	'image_size': 256,
	'g_conv_dim': 128,
	'd_conv_dim': 128,
	'g_repeat_num': 6,
	'd_repeat_num': 6,
	'lambda_cls': 1,
	'lambda_rec': 10,
	'lambda_gp': 10,
	'batch_size': 16,
	'num_iters': 200000,
	'num_iters_decay': 5000,
	'g_lr': 0.0002,
	'd_lr': 0.0004,
	'n_critic': 5,
	'beta1': 0.5,
	'beta2': 0.999,
	'resume_iters': 0, (몇번째 iter의 checkpoint를 받아올지)
	'test_iters': 300000
	'num_workers': 1,
	'mode': 'train',
	'use_tensorboard': True,
	'image_dir': 'data/train', (train 데이터 경로)
	'log_dir': 'data/logs',
	'model_save_dir': 'data/models',
	'sample_dir': 'data/samples', (sample 결과를 저장할 경로)
	'sample_label_dir': 'data/train', (label 정보가 들어있는 경로)
	'result_dir': 'data/results', (test 결과를 저장할 경로)
	'image_path': 'data/test', (test 데이터 경로)
	'log_step': 10,
	'sample_step': 1000,
	'model_save_step': 10000,
	'lr_update_step': 1000,
}
```

### 1) Data preprocessing

```
(env_name) ~/stargan> python datapreprocess.py
```

train폴더의 각 라벨별 이미지들을 얼굴만 crop하여 512로 리사이즈하여 저장하는 전처리 과정을 거친다.

512\*512로 preprocessing된 이미지를 train512폴더에 저장한다.

### 2) Train (실제 학습 시 사용된 command)

```
(env_name) ~/stargan> python main.py --image_dir data/train512 --image_size 256 --resume_iters 660000 --g_lr 0.0001 --d_lr 0.0001 --num_iters 800000 --num_iters_decay 10000 --lr_update_step 10000 --model_save_step 10000 --sample_step 500 --g_conv_dim 128 --d_conv_dim 128
```

### 3) Test

- cli를 활용한 test

```
(env_name) ~/stargan> python3 main.py --mode test --image_size 256 --g_conv_dim 128 --d_conv_dim 128 --test_iters 660000
```

- streamlit을 활용한 test (streamlit 설치 필요)

```jsx
(env_name) ~/stargan> streamlit run streamlit_run.py
```

- flutter 앱을 활용한 test (flutter & fastAPI 환경 구성 필요)

  - fastAPI 백엔드 세팅

  ```bash
  (env_name) ~/stargan> pip install fastapi==0.74.1

  // 필요 시 포트 열어주기
  (env_name) ~/stargan> sudo ufw allow 3000

  (env_name) ~/stargan> uvicorn backend:app --reload --host=0.0.0.0 --port=3000
  ```

  - flutter 앱 세팅 (vsCode 또는 Android Studio 사용, [링크](https://github.com/seoin0110/StarGAN-FE_App) 참고)

  ```
  git clone https://github.com/seoin0110/StarGAN-FE_App.git
  cd StarGAN-FE_App

  flutter pub get
  flutter run lib/main.dart
  ```

## IV. Requirements

```
absl-py==2.1.0
attrs==23.2.0
charset-normalizer==3.3.2
cycler==0.11.0
dlib==19.24.2
facenet-pytorch==2.5.3
filelock==3.12.2
flatbuffers==23.5.26
fonttools==4.38.0
fsspec==2023.1.0
huggingface-hub==0.16.4
idna==3.6
importlib-metadata==6.7.0
kiwisolver==1.4.5
logger==1.4
matplotlib==3.5.3
mediapipe==0.9.0.1
numpy==1.21.6
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
opencv-contrib-python==4.9.0.80
opencv-python==4.9.0.80
packaging==23.2
Pillow==9.5.0
protobuf==3.20.3
pyparsing==3.1.1
python-dateutil==2.8.2
python-xlib==0.33
pytorch-fid==0.3.0
pytorch-fid-wrapper==0.0.4
PyUserInput==0.1.11
PyYAML==6.0.1
requests==2.31.0
safetensors==0.4.2
scipy==1.7.3
six==1.16.0
timm==0.9.12
torch==1.13.1
torchvision==0.14.1
tqdm==4.66.1
typing_extensions==4.7.1
urllib3==2.0.7
zipp==3.15.0
```

## V. References

- [StarGAN-TA](https://github.com/shp216/StarGAN-TA)
- [StarGAN](https://github.com/yunjey/stargan)
- [한국인 감정인식을 위한 복합 영상 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=82)
