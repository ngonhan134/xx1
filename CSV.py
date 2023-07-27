import os
import numpy as np
import cv2
import LMTRP
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time
path = 'DATA3/'
labels = os.listdir(path)
print("Tổng số file trong DATA_SET: ", len(labels))

X = []
y = []
total_time = 0 # Tổng thời gian trích đặc trưng
for i, label in enumerate(labels):
    img_filenames = os.listdir('{}{}/'.format(path, label))
    for filename in tqdm(img_filenames, desc='Processing ' + label):
        filepath = '{}{}/{}'.format(path, label, filename)
        img = cv2.imread(filepath)
        img=    cv2.resize(img,(64,64))
        
        # Ignore if not found face in image
        try:
            start_time = time.time() # Thời điểm bắt đầu trích xuất đặc trưng
            encode = LMTRP.LMTRP_process(img)
            end_time = time.time() # Thời điểm kết thúc trích xuất đặc trưng
            elapsed_time = end_time - start_time # Thời gian trích xuất đặc trưng của ảnh hiện tại
            total_time += elapsed_time # Cộng thời gian trích xuất đặc trưng của ảnh hiện tại vào tổng thời gian
            # print(encode)
        except Exception as e:
            print(e, ":", label, filename)
            continue
        
        X.append(encode)
        y.append(i)

print("Tổng thời gian trích đặc trưng: ", total_time) # Tổng thời gian trích đặc trưng của toàn bộ quá trình

# Convert the image data to numpy arrays
X = np.asarray(X)
y = np.asarray(y)
X = X.reshape(X.shape[0], -1)

# Tạo DataFrame từ mảng X và thêm cột nhãn y
df = pd.DataFrame(X)
df = df.assign(label=y)

# Lưu DataFrame vào file CSV với cả cột chỉ mục
df.to_csv('new.csv')