# -*- coding: utf-8 -*-
"""
Created on Tue May 29 03:08:36 2018
DTI concussion classification: general control vs moderate TBI with FA and MD
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

x = loadmat('modTBI_genctrl.mat')
cind = x['label']
data = x['FA_data']

Labels=cind
n_splits=24
n_trial=1

RF_Performance=np.zeros([n_splits,1])
RF_Accuracy=np.zeros([n_trial,1])
RF_Sensitivity=np.zeros([n_trial,1])
RF_Specificity=np.zeros([n_trial,1])
def perf_measure(y_actual, y_hat, TP, FP, TN, FN):

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==1 and y_actual!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==0 and y_actual!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN) 
    
loo = LeaveOneOut()
for loop in range(0,1):
    index=0
    Y_Predict=np.zeros([1,24])
    Y_True=np.zeros([1,24])
#    Labels=np.reshape(cind_label[loop,:],(24,1))
    for train, test in loo.split(Labels):
        print("%s %s" % (train, test))
        X_train, X_test, Y_train, Y_test = data[train], data[test], Labels[train], Labels[test]
        clf2 = RandomForestClassifier(45, max_depth=64, max_features='sqrt', min_samples_split=2)
        clf2.fit(X_train, Y_train)
        
        Y_Predict[:,index]=clf2.predict(X_test)[0]
        Y_True[:,index]=Y_test
        
        RF_Performance[index]=clf2.score(X_test, Y_test)
        index=index+1
    tn, fp, fn, tp = confusion_matrix(Y_True.T, Y_Predict.T).ravel()
    RF_Sensitivity[loop]=tp/float(tp+fn)
    RF_Specificity[loop]=tn/float(tn+fp)
    RF_Accuracy[loop]=np.mean(RF_Performance,axis=0)    
confi_Sen=[np.percentile(RF_Sensitivity,2.5),np.percentile(RF_Sensitivity,97.5)]
confi_Spe=[np.percentile(RF_Specificity,2.5),np.percentile(RF_Specificity,97.5)] 
confi_acc=[np.percentile(RF_Accuracy,2.5),np.percentile(RF_Accuracy,97.5)]       


