from __future__ import absolute_import, division, print_function

from fastapi import FastAPI
from typing import List
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
from pydantic import BaseModel
import base64
from io import BytesIO
import json
from datetime import datetime
from model import Generator
import os
from PIL import Image, UnidentifiedImageError
from datetime import datetime
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from face_detection import detection_face_test, detection_and_resize_original, get_face_mesh
from torchvision.transforms.functional import to_pil_image
import logging
import time
from facenet_pytorch import MTCNN
import timm

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
logging.basicConfig(level=logging.INFO)
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

app = FastAPI()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=6):
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)
        c_trg_list.append(c_trg.to("cpu"))
    return c_trg_list


class Item(BaseModel):
    image: str

class ImagesItem(BaseModel):
    images: List[str]
    #labels: List[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진을 허용하려면 ["*"]을 사용합니다.
    allow_credentials=True,
    allow_methods=["*"],    # 모든 HTTP 메소드를 허용합니다.
    allow_headers=["*"],    # 모든 헤더를 허용합니다.
)

@app.get("/")
async def root():
    print("get request received at /")
    logging.info("get request received at /")
    return {"message": "Hello World"}


@app.post("/generate")
async def generate_emotion(item: Item):
    print("POST request received at /generate")
    logging.info("POST request received at /generate")
    logging.info("Request received at /generate")  # 요청이 들어왔을 때 로그 남기기

    print("Request received at /generate", flush=True)
    print("ADFADAA")
    # 생성모델 초기 세팅
    labels = ['원본', '분노', '공포', '기쁨', '슬픔', '놀람']
    c_dim = 5
    conv_dim = 128
    image_size = 256
    test_iters='best'
    model_save_dir = f'data/models'
    # G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(test_iters))
    G_path = os.path.join(model_save_dir, 'best-G_900000.ckpt')
    saved_checkpoint_G = torch.load(G_path, map_location=torch.device(device))
    G = Generator(conv_dim, c_dim, 6)
    G.to(device)
    G.load_state_dict(saved_checkpoint_G, strict = False)
    images_data = {}
    try:
        image_data = base64.b64decode(item.image)
        image = Image.open(BytesIO(image_data))
        img_array = np.array(image)
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_list = detection_and_resize_original(img_array_bgr, image_size)
        if len(img_list) == 0:
            return {"massage": "이미지에 얼굴이 인식되지 않습니다."}
        img_file = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2RGB)
        transform = []
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        original = transform(img_file)
        img, (x, y, w, h) = img_list[1]
        image_size = img.size[0] #256
        x_real = transform(img)
        print(x_real.shape)
        x_real = x_real.view(1, 3, image_size, image_size)
        c_org = torch.Tensor([3])
        
        with torch.no_grad():
            x_real = x_real.to(device)
            c_trg_list = create_labels(c_org, c_dim)
            x_fake_list = [x_real]
            x_origin_list = [original]
            x_mesh_list = [original]

            for c_trg in c_trg_list:
                x_fake_list.append(G(x_real, c_trg))
                x_origin_list.append(torch.tensor(original))
                x_mesh_list.append(torch.tensor(original))
            images_data['images'] = []
            for i, fake in enumerate(x_fake_list):
                translate_img = fake.data.cpu().squeeze(0)
                translate_img = translate_img.permute(1, 2, 0).numpy()
                translate_img = cv2.resize(translate_img,(w, h))
                translate_img = torch.from_numpy(translate_img).permute(2, 0, 1) # C, H, W
                min_value = translate_img.min()
                max_value = translate_img.max()
                for j in range(3):
                    for k in range(y, y+h):
                        x_origin_list[i][j][k][x:x+w] = translate_img[j][k-y] # -1에서 1 사이
                face_dict, mesh_img = get_face_mesh(to_pil_image(0.5*x_origin_list[i] + 0.5))
                if face_dict == None:
                    numpy_image = denorm(x_origin_list[i].data.cpu()).numpy()
                    numpy_image = np.transpose(numpy_image, (1, 2, 0))
                    pil_image = Image.fromarray((numpy_image*255).astype(np.uint8))
                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG")
                    img_byte = buffered.getvalue()
                    img_base64 = base64.b64encode(img_byte)
                    
                    img_str = img_base64.decode('utf-8')
                    images_data['images'].append(img_str)
                    continue
                mesh_tensor = transform(mesh_img)
                for j in range(3):
                    for y1, x_list in face_dict.items():
                        if len(x_list) == 1:
                            continue
                        x1, x2 = x_list
                        x_mesh_list[i][j][y1][x1:x2] = mesh_tensor[j][y1][x1:x2]
                numpy_image = denorm(x_mesh_list[i].data.cpu()).numpy()
                numpy_image = np.transpose(numpy_image, (1, 2, 0))
                print(numpy_image.shape)
                pil_image = Image.fromarray((numpy_image*255).astype(np.uint8))
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_byte = buffered.getvalue()
                
                img_base64 = base64.b64encode(img_byte)
                
                img_str = img_base64.decode('utf-8')
                images_data['images'].append(img_str)
        images_data['labels'] = labels
        return JSONResponse(content=images_data, media_type="application/json; charset=utf-8")
          
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/submit")
async def submit_emotion(item: ImagesItem):
    labels = ['original', 'angry', 'fearful', 'happy', 'sad', 'surprised']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{current_time}.jpg"
    for idx, image_base64 in enumerate(item.images):
        if idx == 0:
            continue
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            image.save(f"data/extended_train/{labels[idx]}/{filename}")
        except (ValueError, UnidentifiedImageError) as e:
            return JSONResponse(status_code=400, content={"message": f"이미지 처리 중 오류가 발생했습니다: {e}"})
        except IOError as e:
            return JSONResponse(status_code=500, content={"message": f"이미지 저장 중 오류가 발생했습니다: {e}"})

    return {"massage": "이미지가 저장되었습니다."}