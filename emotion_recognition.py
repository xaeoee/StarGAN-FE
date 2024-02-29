from PIL import Image
from torchvision import transforms as T
import numpy as np
import os
import logging
import time
import torch
import urllib.request
def get_model_path(model_name):
    model_file=model_name+'.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    print(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/'+model_file+'?raw=true'
        print('Downloading',model_name,'from',url)
        urllib.request.urlretrieve(url, fpath)
    return fpath       

class HSEmotionRecognizer:
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b2_7',device='cpu'):
        self.device=device
        #self.idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
        self.idx_to_class={0: 'Anger', 1: 'Fear', 2: 'Happiness', 3: 'Sadness', 4: 'Surprise'}
        self.img_size=260
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
        print(path,self.test_transforms)
    
    def get_probab(self, features):
        x=np.dot(features,np.transpose(self.classifier_weights))+self.classifier_bias
        # print("min_features", np.min(features))
        # print("max_features", np.max(features))
        # print("min_weights", np.min(self.classifier_weights))
        # print("max_weights", np.max(self.classifier_weights))
        # print("x_min: ", np.min(x))
        # print("x_max: ", np.max(x))
        # print("classifier_bias: ", self.classifier_bias)
        return x
    
    def extract_features(self,face_img):
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features=features.data.cpu().numpy()
        return features
        
    def predict_emotions(self,face_img):
        features=self.extract_features(face_img)
        scores=self.get_probab(features)[0]
        scores = np.concatenate((scores[:1], scores[2:4], scores[5:])) #중립 없앰
        x = scores
        pred=np.argmax(x)
        e_x = np.exp(x - np.max(x)[np.newaxis])
        e_x = e_x / e_x.sum()[None]
        return self.idx_to_class[pred],e_x