from ROI2 import *
import os
import LMTRP
import joblib
import numpy as np
import cv2
import glob
from PIL import Image 
import time
import ControlDoor
import cv2

def prediction():
    print("Đã nhận")
    # Đường dẫn tới thư mục chứa các ảnh
    path_out_img = './ROI1'

    # Xóa toàn bộ tệp tin ảnh trong thư mục path_out_img
    file_list = glob.glob(os.path.join(path_out_img, '*.bmp'))
    for file_path in file_list:
        os.remove(file_path)

    roiImageFromHand(path_out_img, option=1, cap=cv2.VideoCapture(0))

    # Lọc ra danh sách các ảnh trong folder
    image_list = glob.glob(os.path.join(path_out_img, '*.bmp'))

    # Load mô hình đã được train
    recognizer = joblib.load('./data1/classifiers/user_classifier.joblib')

    pred = 0
    print_flag = True
    img1_path = './ROI1/0001_0009.bmp'
    img1 = cv2.imread(img1_path)
    # img1 = cv2.resize(img1, (64, 64))
    feature = LMTRP.LMTRP_process(img1)
    feature = feature.reshape(1, -1)
    predict = recognizer.predict_proba(feature)
    print(predict)
    user_prob = predict[0][1]

    if print_flag:
        print("Đang dự đoán...!")
        print_flag = False

    if user_prob > 0.8:
        print(user_prob)
        print("Hợp lệ !!")
        return True
        
    else:
        print(user_prob)
        print('Không xác định')
        return False


# prediction()