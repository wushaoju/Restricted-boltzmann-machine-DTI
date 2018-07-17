# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:15:51 2018

Generating batch data for validation and testing
@author: Shaoju Wu
"""
import numpy as np
from scipy.io import loadmat
import os
import random
from keras.utils import np_utils

def processList(mTBIlist,genctrllist,non_concuss,concuss,concuss_path, nonconcuss_path):
    batchList = list()
    num = len(concuss)+len(non_concuss)
    batchLabel = np.zeros([1,num])
    for i in range(len(concuss)):
        batchList.append(concuss_path+mTBIlist[concuss[i]])
        batchLabel[:,i] = 1 
    for j in range(len(non_concuss)):
        batchList.append(nonconcuss_path+genctrllist[non_concuss[j]]) 
        
    return batchList, batchLabel

def processBatch(batch_List,batch_Label):
    num = len(batch_List)
    batchIm = np.zeros((num,256,256,256,1),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        Im = loadmat(batch_List[i]) 
        batchIm[i][:][:][:][:] = Im['MD_data'].reshape((256, 256, 256,1))
        labels[i] = batch_Label[i]
    return batchIm,labels 

def trainGenerator(train_non_concuss, train_concuss, batch_size, num_classes, concuss_path, nonconcuss_path):
#    mylist = range(3)
    mTBIlist = os.listdir(concuss_path)
    genctrllist = os.listdir(nonconcuss_path)
    batchList,batchLabel = processList(mTBIlist,genctrllist,train_non_concuss,train_concuss,concuss_path, nonconcuss_path)
    new_line = []
    Label_line = []
    index = [n for n in range(len(batchList))]
    random.shuffle(index)
    for m in range(len(batchList)):
        new_line.append(batchList[index[m]])
        Label_line.append(int(batchLabel[:,index[m]]))
    for i in range(int(84 / batch_size)):
        a = i * batch_size
        b = (i + 1) * batch_size
        x_train,x_labels = processBatch(new_line[a:b],Label_line[a:b])
        y = np_utils.to_categorical(np.array(x_labels), num_classes)
#        Im = loadmat(img_path+file)
#        Im = Im['MD_data']
        yield x_train, y, i
        
def valGenerator(val_non_concuss, val_concuss, batch_size, num_classes, concuss_path, nonconcuss_path):
#    mylist = range(3)
    mTBIlist = os.listdir(concuss_path)
    genctrllist = os.listdir(nonconcuss_path)
    batchList,batchLabel = processList(mTBIlist,genctrllist,val_non_concuss,val_concuss,concuss_path, nonconcuss_path)
    new_line = []
    Label_line = []
    index = [n for n in range(len(batchList))]
    random.shuffle(index)
    for m in range(len(batchList)):
        new_line.append(batchList[index[m]])
        Label_line.append(int(batchLabel[:,index[m]]))
    for i in range(int(20 / batch_size)):
        a = i * batch_size
        b = (i + 1) * batch_size
        x_val,x_labels = processBatch(new_line[a:b],Label_line[a:b])
        y_val = np_utils.to_categorical(np.array(x_labels), num_classes)
#        Im = loadmat(img_path+file)
#        Im = Im['MD_data']
        yield x_val, y_val, i        

mildTBI_file = '/media/ycai2/Jilab_drive/3D_CNN/3D_CNN-keras_DTI/MNI_space/matFile/mildTBI/md/visit_1/'
genctrl_file = '/media/ycai2/Jilab_drive/3D_CNN/3D_CNN-keras_DTI/MNI_space/matFile/genctrl/md/visit_1/'
np.random.seed(0)
non_concuss_list = np.random.choice(41, 41, replace=False)
concuss_list = np.random.choice(63, 63, replace=False)
train_non_concuss = non_concuss_list[0:33]
train_concuss = concuss_list[0:51]

val_non_concuss = non_concuss_list[33:41]
val_concuss = concuss_list[51:63]
#non_concuss = np.random.randint(2, size=10)
batch_size = 2
num_classes = 2
concuss_img_path = mildTBI_file
nonconcuss_img_path = genctrl_file
mygenerator = valGenerator(val_non_concuss, val_concuss, batch_size,num_classes,concuss_img_path,nonconcuss_img_path)
index=0
for i in mygenerator:
    aaa,bbb,ccc = i
    index = index +1
    print(ccc)
    
#print(mygenerator) 