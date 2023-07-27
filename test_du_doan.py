import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

import LMTRP
import cv2


recognizer = joblib.load("./data1/classifiers/SVM_classifier.joblib")


img1_path = './ROI1/77.bmp'
img1 = cv2.imread(img1_path)


feature = LMTRP.LMTRP_process(img1)
feature = feature.reshape(1, -1)


predict = recognizer.predict(feature)

print(predict)