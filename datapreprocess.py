import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import math
from tqdm import tqdm

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def image_preprocess_512(file, emotion, idx):
    img_array = np.fromfile(file, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #image = cv2.imread(file)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        
        annotated_image = image.copy()
        if results.detections is None:
            return
        for detection in results.detections:
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
            annotated_image = image[max(y-50, 0):y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (512, 512))

        f_name = file.split(".")[-1]
        if not os.path.exists("data512/" + emotion):
            os.mkdir("data512/" + emotion)
        file_name = f'data512/{emotion}/image{idx}.{f_name}'
        cv2.imwrite(file_name, output_image)


def image_resize_512(file):
    image = cv2.imread(file)    
    output_image = cv2.resize(image, (512, 512))
    print("image_resize_512 is complete")
    f_name = (file.split("/")[-1]).split(".")[0]    
    cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_512.jpg", output_image)
    print("Successfully write file")

#################################################################################
################################ main_part ######################################
#################################################################################

# file = "/home/mineslab-ubuntu/stargan/Original_jpg/robot3.jpg"
train_data_path = "/home/mineslab-ubuntu/stargan/Korean/reduced_Korean"
# #image_resize_256(file)
sad = glob.glob(os.path.join(train_data_path, 'sad') + '/*')
angry = glob.glob(os.path.join(train_data_path, 'angry') + '/*')
happy = glob.glob(os.path.join(train_data_path, 'happy') + '/*')
neutral = glob.glob(os.path.join(train_data_path, 'neutral') + '/*')
surprised = glob.glob(os.path.join(train_data_path, 'surprised') +"/*")
fearful = glob.glob(os.path.join(train_data_path, 'fearful') + '/*')


# train_img_list_resize_256(sad, 'sad')
# train_img_list_resize_256(angry, 'angry')
# train_img_list_resize_256(happy, 'happy')
# train_img_list_resize_256(neutral, 'neutral')
# train_img_list_resize_256(surprised, 'surprised')
# train_img_list_resize_256(fearful, 'fearful')

# print(len(sad), len(angry), len(happy), len(neutral))
# f_name = os.getcwd() + "/data/cartoon/"
# fearful = glob.glob(f_name)
# image_list_preprocess_256(f_name)

# file = "/home/mineslab-ubuntu/stargan/Original_jpg/sakura_ani2.png"
# image_resize_256(file)

#bts = "/home/mineslab-ubuntu/stargan/Original_jpg/sakura_ani2.png"
#img_path = "/home/mineslab-ubuntu/stargan/Original_jpg"
emotion = ['불안', '슬픔', '기쁨', '당황', '분노']
# e_emo : label 폴더 명 
e_emo = ['fearful', 'sad', 'happy', 'surprised', 'angry']
import sys

print(sys.getfilesystemencoding())
img_file = glob.glob("/*")
for emo, e in zip(emotion, e_emo):
    file = f'{e}/*'
    img_file = glob.glob(file)
    print(len(img_file), file, emo, e)
    idx = 1
    for i in tqdm(img_file):
        if i.split('.')[-1] != 'jpg':
            continue
        image_preprocess_512(i, e, idx)
        idx += 1
