# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 01:51:23 2018
DTI concussion classification: general control(77 cases) vs mild TBI(130 cases) 
with FA, MD, AD, RD four modailty
Using Restricted-boltzmann machine with cross validation
@author: Shaoju Wu
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import os

def data_normalization(Im):
    Im_cpy = Im.copy()
    Im_max = Im_cpy.max(axis=1)
    Im_cpy = Im_cpy.T
    Im_cpy /= Im_max
    Im_cpy = Im_cpy.T
    return Im_cpy

def data_demean(x,x_val):
    x_cpy = x.copy()
    x_val_cpy = x_val.copy()
    X_s_mean = x_cpy.mean(axis=0)
    x_cpy -= X_s_mean
    x_val_cpy -= X_s_mean
    return x_cpy, x_val_cpy, X_s_mean     

def make_GBRBM_kfold(train_data, valid_data, Image_type, k, save_path):
 ##############% K-fold based restricted boltzmann machine for training
    # train the model
    gbrbm = GBRBM(n_visible=train_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
    train_data, valid_data, data_mean = data_demean(train_data,valid_data)
    errs, errs_val = gbrbm.fit(train_data, valid_data, n_epoches=640, batch_size=20)
    foldName = str(k)+'_fold/'+Image_type+'/'
    createFolder(save_path+foldName)
    plt.plot(errs)
    plt.show()
    plt.savefig(save_path+foldName+'train.png')
    plt.plot(errs_val)
    plt.savefig(save_path+foldName+'val.png')
    plt.show()
    np.save(save_path+foldName+'data_mean.npy', data_mean) 
    gbrbm.save_weights(filename = save_path+foldName, name = Image_type+'_model') 
    
def make_GBRBMtransform_kfold(train_data, valid_data, Image_type, k, save_path):
 ##############% Transform the data from the first layer with Gaussian kernal RBM
    gbrbm = GBRBM(n_visible=train_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
    train_data, valid_data, data_mean = data_demean(train_data,valid_data)
    foldName = str(k)+'_fold/'+Image_type+'/'   
    gbrbm.load_weights(filename = save_path+foldName, name = Image_type+'_model')
    # transform the training and testing dataset
    transform_data_train = np.zeros([train_data.shape[0] , 1000]).astype(np.float32)
    for i in range(0,train_data.shape[0]):
        transform_data_train[i,:] = gbrbm.transform(train_data[i,:].reshape(1,-1))
        
    transform_data_val = gbrbm.transform(valid_data)      
    return transform_data_train, transform_data_val

def make_BBRBM_kfold(train_data, valid_data, Image_type, k, save_path):     
    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)    
    

##############% path and name of the data FA   
data_path = '../data/'
data_name = 'genctrl_mildTBI_TBSS.mat'
X_data = loadmat(data_path+data_name)
FA_data = X_data['FA_data'].astype(np.float32)
MD_data = X_data['MD_data'].astype(np.float32)
AD_data = X_data['AD_data'].astype(np.float32)
RD_data = X_data['RD_data'].astype(np.float32)

############## normalize MD,AD and RD image to scale (0,1)
MD_norm = data_normalization(MD_data)
AD_norm = data_normalization(AD_data)
RD_norm = data_normalization(RD_data)

############### Concatenate the FA, MD, AD, RD
FA_MD_data = np.concatenate((FA_data, MD_norm), axis=1)
FA_AD_data = np.concatenate((FA_data, AD_norm), axis=1)
FA_RD_data = np.concatenate((FA_data, RD_norm), axis=1)

FA_MD_AD_data = np.concatenate((FA_MD_data, AD_norm), axis=1)
FA_MD_RD_data = np.concatenate((FA_MD_data, RD_norm), axis=1) 

FA_MD_AD_RD_data = np.concatenate((FA_MD_AD_data, RD_data), axis=1)     
      
Label = X_data['label']

n_splits = 10
n_trial = 1
#RBM_Performance = np.zeros([n_splits,1])
RBM_Accuracy = np.zeros([n_splits,1])
RBM_Sensitivity = np.zeros([n_splits,1])
RBM_Specificity = np.zeros([n_splits,1])

SVM_Accuracy = np.zeros([n_splits,1])
SVM_Sensitivity = np.zeros([n_splits,1])
SVM_Specificity = np.zeros([n_splits,1])

Total_mean_acc = np.zeros([n_trial,1])
Total_mean_Sen = np.zeros([n_trial,1])
Total_mean_Spe = np.zeros([n_trial,1])

#%% Stratified K-FOLD cross validation
#skf = StratifiedKFold(n_splits=5)
save_path = '../models/rbm_cross_va/'+str(n_splits)+'/'
createFolder(save_path)

kf = KFold(n_splits=10, random_state=1, shuffle=True)
for loop in range(0,n_trial):
    index=0 
    for train, test in kf.split(Label):

        print("TRAIN:", train, "TEST:", test)
        
############### Concatenate the FA, MD, AD, RD        
        X_train, X_test, Y_train, Y_test = MD_norm[train], MD_norm[test], Label[train], Label[test]
                
        with tf.Graph().as_default():
            make_GBRBM_kfold(X_train, X_test, 'MD', index, save_path)
            
        with tf.Graph().as_default():
             X_train_t, X_test_t = make_GBRBMtransform_kfold(X_train, X_test, 'MD', index, save_path)
        clf2 = svm.SVC(kernel='linear',probability=True)     
#        clf2 = RandomForestClassifier(45, max_depth=5, max_features=None, min_samples_split=2)
        clf2.fit(X_train_t, Y_train)
        SVM_Accuracy[index] = clf2.score(X_test_t, Y_test)
        Y_predict = clf2.predict(X_test_t)
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_predict).ravel()
        SVM_Sensitivity[index] = np.float(tp)/np.float(tp+fn)
        SVM_Specificity[index] = np.float(tn)/np.float(tn+fp)
          
        X_train, X_test, Y_train, Y_test = FA_data[train], FA_data[test], Label[train], Label[test]
        with tf.Graph().as_default():
             X_train_t, X_test_t = make_GBRBMtransform_kfold(X_train, X_test, 'FA', index, save_path)
             
        with tf.Graph().as_default():
            make_GBRBM_kfold(X_train, X_test, 'FA', index, save_path)

        X_train, X_test, Y_train, Y_test = AD_norm[train], AD_norm[test], Label[train], Label[test]
        with tf.Graph().as_default():
            make_GBRBM_kfold(X_train, X_test, 'AD', index, save_path)

        X_train, X_test, Y_train, Y_test = RD_norm[train], RD_norm[test], Label[train], Label[test]
        with tf.Graph().as_default():
            make_GBRBM_kfold(X_train, X_test, 'RD', index, save_path)
        
            
        index=index+1

    Total_mean_acc[loop] = np.mean(SVM_Accuracy,axis=0)
    Total_mean_Sen[loop] = np.mean(SVM_Sensitivity,axis=0)
    Total_mean_Spe[loop] = np.mean(SVM_Specificity,axis=0)

Final_mean_acc=np.mean(Total_mean_acc,axis=0)
Final_mean_Sen=np.mean(Total_mean_Sen,axis=0)
Final_mean_Spe=np.mean(Total_mean_Spe,axis=0)         
        