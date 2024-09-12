from __future__ import absolute_import, division, print_function

from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import streamlit as st
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
import urllib.request

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
logging.basicConfig(level=logging.INFO)
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

# 생성모델
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
    

# 감정 인식 모델
def detect_face(frame):
  bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
  if bounding_boxes is not None:
    bounding_boxes = bounding_boxes[probs > 0.9]
    return bounding_boxes
  return None

def get_model_path(model_name):
    model_file=model_name+'.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/'+model_file+'?raw=true'
        print('Downloading',model_name,'from',url)
        urllib.request.urlretrieve(url, fpath)
    return fpath        
    

class HSEmotionRecognizer:
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b2_8_best',device='cpu'):
        self.device=device
        self.is_mtl='_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class={0: 'Anger', 1: 'Fear', 2: 'Happiness', 3: 'Neutral', 4: 'Sadness', 5: 'Surprise'}
        else:
            self.idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        self.img_size=224 if '_b0_' in model_name else 260
        self.test_transforms = T.Compose(
            [
                T.Resize((self.img_size,self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ]
        )
        
        path=get_model_path(model_name)
        if device == 'cpu':
            model=torch.load(path, map_location=torch.device('cpu'))
        else:
            model=torch.load(path)
        if isinstance(model.classifier,torch.nn.Sequential):
            self.classifier_weights=model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias=model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights=model.classifier.weight.cpu().data.numpy()
            self.classifier_bias=model.classifier.bias.cpu().data.numpy()
        
        model.classifier=torch.nn.Identity()
        model=model.to(device)
        self.model=model.eval()
    
    def get_probab(self, features):
        x=np.dot(features,np.transpose(self.classifier_weights))+self.classifier_bias
        return x
    
    def extract_features(self,face_img):
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features=features.data.cpu().numpy()
        return features
        
    def predict_emotions(self,face_img, logits=True):
        features=self.extract_features(face_img)
        scores=self.get_probab(features)[0]
        scores = np.concatenate((scores[:1], scores[2:]))
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
        pred=np.argmax(x)
        
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2]=e_x
            else:
                scores=e_x
        return self.idx_to_class[pred],scores
        
    def extract_multi_features(self,face_img_list):
        imgs = [self.test_transforms(Image.fromarray(face_img)) for face_img in face_img_list]
        features = self.model(torch.stack(imgs, dim=0).to(self.device))
        features=features.data.cpu().numpy()
        return features
        
    def predict_multi_emotions(self,face_img_list, logits=True):
        features=self.extract_multi_features(face_img_list)
        scores=self.get_probab(features)
        if self.is_mtl:
            preds=np.argmax(scores[:,:-2],axis=1)
        else:
            preds=np.argmax(scores,axis=1)
        if self.is_mtl:
            x=scores[:,:-2]
        else:
            x=scores
        pred=np.argmax(x[0])
        
        if not logits:
            e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:,None]
            if self.is_mtl:
                scores[:,:-2]=e_x
            else:
                scores=e_x

        return [self.idx_to_class[pred] for pred in preds],scores

def main():
    #config
    labels = ['original', 'angry', 'fearful', 'happy', 'sad', 'surprised']
    # 생성모델
    c_dim = 5
    conv_dim = 128
    image_size = 256
    test_iters='best'
    model_save_dir = f'data/models'
    # G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(test_iters))
    G_path = os.path.join(model_save_dir, 'best-G_900000.ckpt')

    saved_checkpoint_G = torch.load(G_path, map_location=torch.device('cpu'))
    G = Generator(conv_dim, c_dim, 6)
    G.to('cpu')
    G.load_state_dict(saved_checkpoint_G, strict = False)
    model_name='enet_b2_7'
    fer = HSEmotionRecognizer(model_name=model_name, device=device)

    feedback = {'Neutral':'입꼬리를 올리고, 눈 웃음을 지어보세요!', 'Fear': '눈 웃음을 지어보세요!', 'Happiness' : '잘했어요!!', 'Contempt' : '입꼬리를 올리고, 눈 웃음을 지어보세요!', 'Anger' : '눈과 눈썹에 힘을 풀고, 입꼬리를 올려보세요!', 'Sadness' : '눈을 반달모양으로 뜨고, 입꼬리를 올려보세요!', 'Disgust' : '눈을 반만 뜨고, 입꼬리를 올려보세요!', 'Surprise' : '눈을 반만 뜨고, 입꼬리를 올려보세요!'}
    st.title('EMOKIDS')
    choice = st.sidebar.selectbox('메뉴', ['감정 생성', '감정 피드백'])
    if choice == '감정 생성':
        st.subheader('이미지 파일 업로드')
        file = st.file_uploader('이미지 파일을 업로드 하세요.', type=['jpg','jpeg', 'png'])
        cols = [*st.columns(len(labels)//2), *st.columns(len(labels)//2)]
        if file is not None:
            current_time = datetime.now()
            file.name = current_time.isoformat().replace(":","_").replace(".","_") + '.jpg'

            start = time.time()

            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_file = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_list = detection_and_resize_original(img_file, image_size)
            img_file = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2RGB)
            transform = []
            transform.append(T.ToTensor())
            transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            transform = T.Compose(transform)
            original = transform(img_file)
            img, (x, y, w, h) = img_list[1]
            image_size = img.size[0] #256
            x_real = transform(img)
            x_real = x_real.view(1, 3, image_size, image_size)
            
            c_org = torch.Tensor([3])
            
            with torch.no_grad():
                x_real = x_real.to("cpu")
                c_trg_list = create_labels(c_org, c_dim)
                x_fake_list = [x_real]
                x_origin_list = [original]
                x_mesh_list = [original]

                for c_trg in c_trg_list:
                    x_fake_list.append(G(x_real, c_trg))
                    x_origin_list.append(torch.tensor(original))
                    x_mesh_list.append(torch.tensor(original))
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
                        with cols[i]:
                            numpy_image = denorm(x_origin_list[i].data.cpu()).numpy()
                            numpy_image = np.transpose(numpy_image, (1, 2, 0))
                            pil_image = Image.fromarray((numpy_image*255).astype(np.uint8))
                            st.image(pil_image, width=250)
                            st.caption(labels[i])
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
                    pil_image = Image.fromarray((numpy_image*255).astype(np.uint8))
                    with cols[i]:
                        st.image(pil_image, width=250)
                        st.caption(labels[i])
                    
                    
            end = time.time()
            st.text(f"{end-start:.5f} sec")
    elif choice == '감정 피드백':
        st.subheader('행복한 표정 짓기')
        run = st.checkbox("Run")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        caption_placeholder = st.empty()
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
            st.session_state.total_time = 0
        while run:
            start = time.time()
            _, frame = camera.read()    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bounding_boxes=detect_face(frame)
            if bounding_boxes is not None and bounding_boxes.any():
                box = bounding_boxes[0].astype(int)
                x1, y1, x2, y2 = box[0:4]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                face_img = frame[y1:y2, x1:x2, :]
                emotion, scores = fer.predict_emotions(face_img, logits=False)
                caption_placeholder.caption(feedback[emotion])
            else:
                caption_placeholder.caption("no face")
            end = time.time()
            st.session_state.total_time += (end-start)
            st.session_state.frame_count += 1
            FRAME_WINDOW.image(frame)
        st.text(f"프레임 수 : {st.session_state.frame_count}")
        if st.session_state.frame_count > 0:
            st.text(f"프레임 당 처리 시간 : {st.session_state.total_time/st.session_state.frame_count:.5f} sec")


                    





        
if __name__ == '__main__':
    main()