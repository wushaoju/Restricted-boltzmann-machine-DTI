# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:41:22 2018
DTI concussion classification: general control(77 cases) vs mild TBI(130 cases) 
with FA, MD, AD, RD four modailty
@author: Shaoju Wu

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from scipy.io import loadmat
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def data_normalization(Im):
    Im_cpy = Im.copy()
    Im_max = Im_cpy.max(axis=1)
    Im_cpy = Im_cpy.T
    Im_cpy /= Im_max
    Im_cpy = Im_cpy.T
    return Im_cpy

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
n_trial = 5
#RF_Performance = np.zeros([n_splits,1])
RF_Accuracy = np.zeros([n_splits,1])
RF_Sensitivity = np.zeros([n_splits,1])
RF_Specificity = np.zeros([n_splits,1])

Total_mean_acc = np.zeros([n_trial,1])
Total_mean_Sen = np.zeros([n_trial,1])
Total_mean_Spe = np.zeros([n_trial,1])

#%% Stratified K-FOLD cross validation
#skf = StratifiedKFold(n_splits=5)
kf = KFold(n_splits=10, random_state=1, shuffle=True)
   
for loop in range(0,n_trial):
    index=0 
    for train, test in kf.split(Label):
        print("TRAIN:", train, "TEST:", test)

        X_train, X_test, Y_train, Y_test = FA_MD_data[train], FA_MD_data[test], Label[train], Label[test]
        clf2 = RandomForestClassifier(45, max_depth=64, max_features='sqrt', min_samples_split=2)
        clf2.fit(X_train, Y_train)
        RF_Accuracy[index]=clf2.score(X_test, Y_test)
        Y_predict = clf2.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_predict).ravel()
        RF_Sensitivity[index] = np.float(tp)/np.float(tp+fn)
        RF_Specificity[index] = np.float(tn)/np.float(tn+fp)
        index=index+1
    Total_mean_acc[loop] = np.mean(RF_Accuracy,axis=0)
    Total_mean_Sen[loop] = np.mean(RF_Sensitivity,axis=0)
    Total_mean_Spe[loop] = np.mean(RF_Specificity,axis=0)
    
Final_mean_acc=np.mean(Total_mean_acc,axis=0)
Final_mean_Sen=np.mean(Total_mean_Sen,axis=0)
Final_mean_Spe=np.mean(Total_mean_Spe,axis=0)    
    