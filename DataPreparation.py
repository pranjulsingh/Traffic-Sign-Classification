import cv2
import os
import glob
from random import shuffle

directories = os.listdir('Images/')

#deleting unnecessary csv files
for directory in directories:
    location = 'Images/'+directory+'/*.csv'
    pth = glob.glob(location)
    for file in pth:
        print(file)
        os.remove(file)
print('all csv removed')

#splitting train test and validate dataset in saperate folder
k=0
for directory in directories:
    location = 'Images/'+directory+'/'
    files = os.listdir(location)
    shuffle(files)
    test_files = files[:int(len(files)/10)]
    validate_files = files[int(len(files)/10):int(len(files)/10+len(files)/10)]
    train_files = files[int(len(files)/10+len(files)/10):]
    i=0
    os.makedirs('dataset/test/'+str(k))
    os.makedirs('dataset/train/'+str(k))
    os.makedirs('dataset/validate/'+str(k))
    for file in test_files:
        img = cv2.imread('Images/'+directory+'/'+file)
        print(file)
        cv2.imwrite('dataset/test/'+str(k)+'/'+str(i)+'.jpg',img)
        i+=1
    i=0
    for file in validate_files:
        img = cv2.imread('Images/'+directory+'/'+file)
        print(file)
        cv2.imwrite('dataset/validate/'+str(k)+'/'+str(i)+'.jpg',img)
        i+=1
    i=0;
    for file in train_files:
        img = cv2.imread('Images/'+directory+'/'+file)
        print(file)
        cv2.imwrite('dataset/train/'+str(k)+'/'+str(i)+'.jpg',img)
        i+=1
    k+=1

print('all done')
    
