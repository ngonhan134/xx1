import numpy as np
import os, cv2
import LMTRP
from sklearn.svm import SVC
from joblib import dump 
from sklearn.model_selection import GridSearchCV
import joblib
import glob
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def train_classifer():
    # Read all the images in custom data-set

# Read all the images in custom data-set
    path1 = os.path.join(os.getcwd()+"/data1/user/")  # path to images of authorized users
    path2 = os.path.join(os.getcwd()+"/data1/unknown2/")  # path to images of unauthorized users

    features = []
    labels = []
    num_images = 0

    # Store images in a numpy format and corresponding labels in labels list
    for folder in glob.glob(path1 + '/*'):
        # name = folder.split('/')[-1] # get name of the folder
        for imgpath in glob.glob(folder + '/*.bmp'):
            img = cv2.imread(imgpath)
            # img = cv2.resize(img, (64,64))
            feature = LMTRP.LMTRP_process(img) # extract feature from image
            features.append(feature)
            num_images += 1
            print("Number of images with features extracted:", num_images)
            labels.append(1) # add the name of the folder as label

    for imgpath in glob.glob(path2 + '/*.bmp'):
        img = cv2.imread(imgpath)
        # img = cv2.resize(img, (64,64))
        feature = LMTRP.LMTRP_process(img) # extract feature from image
        features.append(feature)
        num_images += 1
        print("Number of images with features extracted of UNKNOW", num_images)
        labels.append(-1) # label all images in unknown folder as -1 (false)

    features = np.asarray(features)
    labels = np.asarray(labels)
    features = features.reshape(features.shape[0],-1)
    # print(features.shape)
    # print(labels)
    # Define the parameters for SVM
    smote = SMOTE()
    features, labels = smote.fit_resample(features, labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    param_grid = {'C': [0.1,0.5,1,2,10,50,100,300,1000,10000],
                  'gamma': [0.1, 0.15, 0.01, 0.001624, 0.01123, 0.000003134, 0.1, 0.567, 0.9, 0.74845, 0.0456, 0.0076677, 0.001, 0.0001,1],
                  'kernel': ['rbf']}
    
    model = GridSearchCV(SVC(class_weight='balanced',probability=True), cv=5,param_grid=param_grid, n_jobs=-1,verbose=3)
    model.fit(X_train, y_train)
    best_params = model.best_params_
    print("Best hyperparameters: ", best_params)

    # Initialize the SVM model with best hyperparameters
    best_svm = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'],probability=True)

    best_svm.fit(X_train, y_train)

    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)

    # Đánh giá độ chính xác trên training set và test set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Training accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    # Báo cáo kết quả trên test set
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))



#     # y_pred = best_svm.predict(features)
#     y_pred = best_svm.predict(X_test)
#     print(classification_report(y_test, y_pred))
# # Save the trained SVM model
    dump(best_svm,"./data1/classifiers/user_classifier.joblib")

    print("Training completed successfully!")
# train_classifer()