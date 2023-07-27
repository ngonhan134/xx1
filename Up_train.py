import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import LMTRP
import glob

recognizer = load("./data1/classifiers/SVM_classifier.joblib")

print("Các lớp ban đầu:", recognizer.classes_) 

path1 = os.path.join(os.getcwd()+"/data1/user/")


features = []
labels = []
num_images = 0

    # Store images in a numpy format and corresponding labels in labels list
for folder in glob.glob(path1 + '/*'):
        # name = folder.split('/')[-1] # get name of the folder
    for imgpath in glob.glob(folder + '/*.bmp'):
        img = cv2.imread(imgpath)

        feature = LMTRP.LMTRP_process(img) # extract feature from image
        features.append(feature)
        num_images += 1
        print("Number of images with features extracted:", num_images)
        labels.append(21) # add the name of the folder as label



features = np.asarray(features)

features = features.reshape(features.shape[0],-1)



recognizer.warm_start = True
classes = np.append(recognizer.classes_, 21)
recognizer = SVC(classes=classes)
recognizer.fit(features, labels)

dump(recognizer, './data1/classifiers/svc_model_updated.joblib')
print("Các lớp sau khi update:", recognizer.classes_)
