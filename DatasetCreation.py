import cv2
import os
from random import shuffle
import numpy as np

directories = os.listdir('dataset/train/')
X_train = []
y_train = []
for directory in directories:
    location = 'dataset/train/'+directory+'/'
    files = os.listdir(location)
    shuffle(files)
    for file in files:
        img = cv2.imread('dataset/train/'+directory+'/'+file)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_train.append(img)
        y_train.append(int(directory))
    print(directory)
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print('train done')

    
directories = os.listdir('dataset/test/')
X_test = []
y_test = []
for directory in directories:
    location = 'dataset/test/'+directory+'/'
    files = os.listdir(location)
    shuffle(files)
    for file in files:
        img = cv2.imread('dataset/test/'+directory+'/'+file)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test.append(img)
        y_test.append(int(directory))
    print(directory)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape)
print('test done')


directories = os.listdir('dataset/validate/')
X_validate = []
y_validate = []
for directory in directories:
    location = 'dataset/validate/'+directory+'/'
    files = os.listdir(location)
    shuffle(files)
    for file in files:
        img = cv2.imread('dataset/validate/'+directory+'/'+file)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_validate.append(img)
        y_validate.append(int(directory))
    print(directory)
X_validate = np.array(X_validate)
y_validate = np.array(y_validate)
print(X_validate.shape)
print('test done')

np.savez('dataset/final_dataset001',X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_validate=X_validate,y_validate=y_validate)
