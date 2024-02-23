from __future__ import absolute_import, division, print_function

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
from pydantic import BaseModel
import base64
from io import BytesIO
import json

from model import Generator
import os
from PIL import Image
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진을 허용하려면 ["*"]을 사용합니다.
    allow_credentials=True,
    allow_methods=["*"],    # 모든 HTTP 메소드를 허용합니다.
    allow_headers=["*"],    # 모든 헤더를 허용합니다.
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate")
async def generate_emotion(item: Item):
    # 생성모델 초기 세팅
    labels = ['original', 'angry', 'fearful', 'happy', 'sad', 'surprised']
    c_dim = 5
    conv_dim = 128
    image_size = 256
    test_iters='best'
    model_save_dir = f'stargan_new_6_leaky/models/best-model'
    G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(test_iters))
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
                logging.info("생성 모델!!")
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
                    print("face mesh 찾기 실패")
                    numpy_image = denorm(x_origin_list[i].data.cpu()).numpy()
                    numpy_image = np.transpose(numpy_image, (1, 2, 0))
                    pil_image = Image.fromarray((numpy_image*255).astype(np.uint8))
                    # PIL 이미지를 바이트로 변환
                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG")
                    img_byte = buffered.getvalue()
                    
                    # 바이트를 Base64로 인코딩
                    img_base64 = base64.b64encode(img_byte)
                    
                    # Base64 인코딩 문자열을 UTF-8로 디코딩하여 JSON 응답 가능하게 만듦
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
                # PIL 이미지를 바이트로 변환
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_byte = buffered.getvalue()
                
                # 바이트를 Base64로 인코딩
                img_base64 = base64.b64encode(img_byte)
                
                # Base64 인코딩 문자열을 UTF-8로 디코딩하여 JSON 응답 가능하게 만듦
                img_str = img_base64.decode('utf-8')
                images_data['images'].append(img_str)
        images_data['label'] = labels
        return JSONResponse(content=images_data)
          
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    