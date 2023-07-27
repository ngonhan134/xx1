import os
from sklearn.metrics import f1_score,recall_score,precision_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump 

from sklearn.metrics import classification_report, confusion_matrix
path = 'DATA3/'
import matplotlib.pyplot as plt

labels = os.listdir(path)

print("Tổng số file trong DATA_SET: ", len(labels))



data = pd.read_csv('new.csv',index_col=0)
X = data.iloc[:, :-1].values
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train.reshape(X_train.shape[0], -1)


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
param_grid = {'C': [0.1,1,10,50,100,300,1000,10000],
                  'gamma': [0.1, 0.15, 0.01, 0.001624, 0.01123, 0.1, 0.567, 0.9, 0.74845, 0.0456, 0.001, 0.0001,1],
                  'kernel': ['rbf']}
svc_model = svm.SVC()

grid_search = GridSearchCV(svc_model, param_grid, cv=5,verbose=3,n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)

svc_model = svm.SVC(kernel=grid_search.best_params_['kernel'],
                     C=grid_search.best_params_['C'],
                     gamma=grid_search.best_params_['gamma'],
)

svc_model.fit(X_train, y_train)


# Dự đoán nhãn cho tập kiểm tra
y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
# In bảng đánh giá mô hình
print("Classification Report:")
print(classification_report(y_test, y_pred))


dump(svc_model,"./data1/classifiers/SVM_classifier.joblib")





























# from sklearn.metrics import precision_score, recall_score, f1_score

# # Tính toán Precision, Recall và F1-Score cho từng lớp
# precision_scores = [precision_score(y_test, y_pred, average=None)[i] for i in range(len(labels))]
# recall_scores = [recall_score(y_test, y_pred, average=None)[i] for i in range(len(labels))]
# f1_scores = [f1_score(y_test, y_pred, average=None)[i] for i in range(len(labels))]

# # Tính toán Precision, Recall và F1-Score micro-average
# precision_scores_micro = precision_score(y_test, y_pred, average='micro')
# recall_scores_micro = recall_score(y_test, y_pred, average='micro')
# f1_scores_micro = f1_score(y_test, y_pred, average='micro')

# # Vẽ biểu đồ chung
# plt.figure(figsize=(10, 6))
# x = np.arange(len(labels))

# # Vẽ Precision
# plt.bar(x, precision_scores, width=0.2, align='center', label='Precision')
# plt.plot(x, [precision_scores_micro] * len(labels), color='red', linestyle='--')
# plt.text(x[-1] + 0.2, precision_scores_micro, f'{precision_scores_micro:.2f}', va='center')

# # Vẽ Recall
# plt.bar(x + 0.2, recall_scores, width=0.2, align='center', label='Recall')
# plt.plot(x, [recall_scores_micro] * len(labels), color='blue', linestyle='--')
# plt.text(x[-1] + 0.2, recall_scores_micro, f'{recall_scores_micro:.2f}', va='center')

# # Vẽ F1-Score
# plt.bar(x + 0.4, f1_scores, width=0.2, align='center', label='F1-Score')
# plt.plot(x, [f1_scores_micro] * len(labels), color='green', linestyle='--')
# plt.text(x[-1] + 0.2, f1_scores_micro, f'{f1_scores_micro:.2f}', va='center')

# plt.xlabel('Labels')
# plt.ylabel('Scores')
# plt.title('Precision, Recall, and F1-Score for Each Label on Test Set')
# plt.xticks(x + 0.2, labels)
# plt.legend()
# plt.grid(True)
# plt.show()




# Tính toán và vẽ biểu đồ F1 score trên tập test
# f1_score_list = []
# for i in range(len(labels)):
#     f1_score_list.append(f1_score(y_test, y_pred, average=None)[i])
    
# #recall
# recall_score_list = []
# for i in range(len(labels)):
#     recall_score_list.append(recall_score(y_test, y_pred, average=None)[i])

# #precion
# pre_score_list = []
# for i in range(len(labels)):
#     pre_score_list.append(precision_score(y_test, y_pred, average=None)[i])

# plt.figure(figsize=(10,5))
# plt.bar(labels, f1_score_list)
# plt.xlabel("Labels")
# plt.ylabel("F1 Score")
# plt.title("F1 Score on Test Set")
# plt.show()


# In ma trận nhầm lẫn
confusion_mtx = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mtx)

# Tạo đồ thị ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# param_grid = {'C': np.logspace(3, 5, num=20, dtype=float),
#               'gamma': np.logspace(-4, -1, num=20, dtype=float),
#               'kernel': ['rbf']
            #   }

# confusion_mtx = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(confusion_mtx)

# # Tạo đồ thị ma trận nhầm lẫn
# plt.figure(figsize=(8, 6))
# plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.colorbar()

# # Hiển thị số trên đồ thị ma trận nhầm lẫn
# thresh = confusion_mtx.max() / 2.
# for i in range(confusion_mtx.shape[0]):
#     for j in range(confusion_mtx.shape[1]):
#         plt.text(j, i, format(confusion_mtx[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if confusion_mtx[i, j] > thresh else "black")

# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels, rotation=45)
# plt.yticks(tick_marks, labels)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

