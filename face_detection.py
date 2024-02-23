from glob import glob
import glob
import cv2
import mediapipe as mp
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
import os
import time
import datetime
from PIL import Image
from facemesh import get_dlib_face_detector, display_facial_landmarks, align_and_crop_face


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def detection_face_test(image):
    
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        image_list = []
        annotated_image = image.copy()
        try :
            for i, detection in enumerate(results.detections):
                # print('Nose tip:')
                # print(mp_face_detection.get_key_point(
                #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                #print(detection)
                normalized_x=detection.location_data.relative_bounding_box.xmin
                normalized_y=detection.location_data.relative_bounding_box.ymin
                normalized_w=detection.location_data.relative_bounding_box.width
                normalized_h=detection.location_data.relative_bounding_box.height
                x = max(math.floor(normalized_x * image_width), 0)
                y = max(math.floor(normalized_y * image_height), 0)
                w = min(math.floor(normalized_w * image_width), image_width - 1)
                h = min(math.floor(normalized_h * image_height), image_height - 1)
                
                #mp_drawing.draw_detection(annotated_image, detection)
                if w % 4 != 0:
                    w -= w % 4
                if h % 4 != 0:
                    h -= h % 4
                print(x,y,w,h)
                annotated_image = image[y-h//4:y+h, x:x+w]

                cv2.imwrite("data/test/{}.jpg".format(i), annotated_image)
                image_list.append([Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)), (x, y, w, h)])
        except:
            pass
    return image_list

def detection_and_resize_original(image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(image)

        # Draw face detections of each face.
        image_list = []
        image_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        annotated_image = image.copy()
        for i, detection in enumerate(results.detections):
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #print(detection)
            normalized_x=detection.location_data.relative_bounding_box.xmin
            normalized_y=detection.location_data.relative_bounding_box.ymin
            normalized_w=detection.location_data.relative_bounding_box.width
            normalized_h=detection.location_data.relative_bounding_box.height
            x = max(math.floor(normalized_x * image_width), 0)
            y = max(math.floor(normalized_y * image_height), 0)
            w = min(math.floor(normalized_w * image_width), image_width - 1)
            h = min(math.floor(normalized_h * image_height), image_height - 1)
                    
            #mp_drawing.draw_detection(annotated_image, detection)
            
            # if w % 4 != 0:
            #     w -= w % 4
            # if h % 4 != 0:
            #     h -= h % 4
            a_h = 0
            # a_h = 50
            print(a_h)
            resize_face_size = size
            '''
            if w > resize_face_size:
                # resize_w = 188 * image.shape[1] // w
                # resize_h = image.shape[0] * resize_w // image.shape[1]
                image = cv2.resize(image, (0,0), fx = resize_face_size / w, fy = resize_face_size / h, interpolation=cv2.INTER_AREA)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(image.shape)
                break
            '''
            print(x,y,w,h)
            x = x + w // 16
            y = max((y - a_h) + h // 16, 0)
            w = w * 7 // 8
            h = (h + a_h) * 7 // 8
            #annotated_image = image[y-h//4:y+h, x:x+w]
            annotated_image = image[y:y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (size, size))
            #annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            image_list.append([Image.fromarray(output_image), (x, y, w, h)])
            cv2.imwrite("data/{}.jpg".format(i), annotated_image)
            
            # cv2.imwrite("data/trash/original_{}.jpg".format(i), image)
    return image_list # index 0 : resized original img, index 1 ~ : face img


def adjust_forehead(eyebrow_end, chin_start):
    # 이마를 포함하기 위해 y 좌표를 조정
    # 턱선과 눈썹 사이의 거리의 20%를 이마 영역으로 추가
    forehead_height = abs(int((eyebrow_end - chin_start) * 0.2))
    forehead_start_y = max(0, eyebrow_end - forehead_height)

    return forehead_start_y

def get_face_mesh(img):
    
    face_detector = get_dlib_face_detector()
    landmarks = face_detector(img)
        
    try:
        print(landmarks[0].shape)
    except IndexError:
        print("landmarks 리스트가 비어 있거나 인덱스가 범위를 벗어났습니다.")
        return None, None
    fa = landmarks[0][:17]
    min_y = int(min(landmarks[0].T[1]))
    max_y = int(max(landmarks[0].T[1]))
    min_y = adjust_forehead(min_y, max_y)
    ima = np.array(img)
    
    face = {}

    pre_X, pre_Y = fa[0]
    for i, (x, y) in enumerate(fa):
        if i == 0:
            continue
    
        Range = range(pre_Y, y) if pre_Y < y else range(pre_Y, y, -1)
        for fy in Range:
            fx = ((x - pre_X) / (y - pre_Y))*(fy - y) + x
            if fy not in face:
                face[fy] = []
            face[fy].append(round(fx))

        pre_X, pre_Y = x, y
    
    face_ = {key : value for key, value in face.items() if len(face[key]) == 2}
    x_duo = face_[min(face_.keys())]
    for k, v in face.items():
        if k < min(face_.keys()):
            face[k] = x_duo

    white = np.full(shape = ima.shape, fill_value=255, dtype=np.uint8)
 
    for y, x_list in face.items():
        if len(x_list) == 1:
            continue
        x1, x2 = x_list
        white[y][x1 : x2+1] = ima[y][x1 : x2+1]

    m = min(face.keys())
    
    x1, x2 = face[m]
    
    for y in range(min_y, m):
        white[y][x1 : x2+1] = ima[y][x1 : x2+1]
        face[y] = [x1, x2]

    face_only = Image.fromarray(white)
    #face_only.show()
    return face, face_only

#get_face_mesh("Original_jpg/angelina.jpg")

# data = "/home/mineslab-ubuntu/stargan/Original_jpg/tak.jpg"
# data1 = "/home/mineslab-ubuntu/다운로드/bss.jpg"
# imgs = detection_and_resize_original(cv2.imread(data))
# #imgs = detection_face_test(cv2.imread(data))
# print(len(imgs))