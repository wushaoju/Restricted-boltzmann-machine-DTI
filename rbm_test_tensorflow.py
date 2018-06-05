# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:20:21 2018
Testing Restricted Boltzmann Machine in Tensorflow
@author: Shaoju WU
"""
import numpy as np
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data
from scipy.io import loadmat

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

def data_normalization(x):
    x_cpy = x.copy()
    X_s_mean = x_cpy.mean(axis=0)
#    X_s_std = x.std(axis=0)
    x_cpy -= X_s_mean
#    x /= X_s_std

#    normalize_img.shape=x.shape
    return x_cpy   
    
    
def make_BBRBM(Image_type,image_type):
    # path and name of the data    
    data_path = '../data/'
    data_name = 'TBSS_'+Image_type+'_Rawimage_249.mat'        
    X_data = loadmat(data_path+data_name)
    X_data = X_data[Image_type+'_image'].astype(np.float32)
    
    X_s_max = X_data.max(axis=0)
    X_data /= X_s_max
    # separate validation and testing dataset
    n_train=229
    n_val=20
    X_train = X_data[:n_train]
    X_val = X_data[-n_val:]
    
    # train the model
    bbrbm = BBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
    errs,errs_val = bbrbm.fit(X_train, X_val, n_epoches=900, batch_size=20)
    # plot the results
    plt.plot(errs)
    plt.show()
    plt.plot(errs_val)
    plt.show()
    # save the model
    save_path='../models/rbm_'+image_type+'/'
    save_name= image_type+'_model'
    bbrbm.save_weights(filename=save_path, name=save_name) 
    
    return bbrbm

def make_GBRBM(Image_type,image_type):
    # path and name of the data    
    data_path = '../data/'
    data_name = 'TBSS_'+Image_type+'_Rawimage_249.mat'        
    X_data = loadmat(data_path+data_name)
    X_data = X_data[Image_type+'_image'].astype(np.float32)
    
    # normalize the data
    if Image_type != 'FA': 
        X_s_max = X_data.max(axis=0)
        X_data /= X_s_max
        
    X_data = data_normalization(X_data)    
    # separate validation and testing dataset
    n_train=229
    n_val=20
    X_train = X_data[:n_train]
    X_val = X_data[-n_val:]
    
    # train the model
    gbrbm = GBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
    errs,errs_val = gbrbm.fit(X_train, X_val, n_epoches=450, batch_size=20)
    # plot the results
    plt.plot(errs)
    plt.show()
    plt.plot(errs_val)
    plt.show()
    # save the model
    save_path='../models/GBRBM/rbm_'+image_type+'/'
    save_name= image_type+'_model'
    gbrbm.save_weights(filename=save_path, name=save_name) 
    
    return gbrbm

def make_BBRBM_layer2

   
def make_transform(Image_type,image_type):
    data_path = '../data/'
    data_name = 'TBSS_'+Image_type+'_Rawimage_249.mat'
    X_data = loadmat(data_path+data_name)
    X_data = X_data[Image_type+'_image'].astype(np.float32)
    
    # normalize the data
    if Image_type != 'FA': 
        X_s_max = X_data.max(axis=0)
        X_data /= X_s_max
        
    X_data = data_normalization(X_data)    
    # separate validation and testing dataset
    n_train=229
    n_val=20
    X_train = X_data[:n_train]
    X_val = X_data[-n_val:]
    # save the model
    save_path='../models/GBRBM/rbm_'+image_type+'/'
    save_name= image_type+'_model'
    # train the model
    gbrbm = GBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
    # loading weights from the path
    gbrbm.load_weights(filename = save_path, name = save_name)
    # transform training and validation dataset
    transform_data_train = np.zeros([n_train , 1000]).astype(np.float32)
    
    for i in range(0,n_train):
        transform_data_train[i,:] = gbrbm.transform(X_train[i,:].reshape(1,-1))
    
    transform_data_val = gbrbm.transform(X_val)      
    return transform_data_train, transform_data_val    

##%% path and name of the data FA   
data_path = '../data/'
data_name = 'TBSS_FA_Rawimage_249.mat'        
X_data = loadmat(data_path+data_name)
X_data = X_data['FA_image'].astype(np.float32)
# separate validation and testing dataset
n_train=229
n_val=20
X_train = X_data[:n_train]
X_val_original = X_data[-n_val:]
# train the model
#bbrbm = BBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.05, momentum=0.9, use_tqdm=True)
#errs,errs_val = bbrbm.fit(X_train, X_val, n_epoches=350, batch_size=20)
## plot the results
#plt.plot(errs)
#plt.show()
#plt.plot(errs_val)
#plt.show()
## save the model
#save_path='../models/'
#save_name='fa_model'
#bbrbm.save_weights(filename=save_path, name=save_name)
#%%
#md_bbrbm=make_BBRBM('MD','md')
#ad_bbrbm=make_BBRBM('AD','ad')
#rd_bbrbm=make_BBRBM('RD','rd')
#%% Gaussian based Restricted boltzmann machine

#fa_gbrbm = make_GBRBM('FA','fa')
#md_gbrbm = make_GBRBM('MD','md')
#md_gbrbm = make_GBRBM('AD','ad')
#md_gbrbm = make_GBRBM('RD','rd')
#%% first layer data transform 
fa_train, fa_val = make_transform('FA','fa')
md_train, md_val = make_transform('MD','md')
ad_train, ad_val = make_transform('AD','ad')
rd_train, rd_val = make_transform('RD','rd')

#%% concatenate the results from the first layer
temp = np.concatenate((fa_train, md_train), axis=1)
temp = np.concatenate((temp, ad_train), axis=1)
train_layer1 = np.concatenate((temp, rd_train), axis=1)

temp1 = np.concatenate((fa_val, md_val), axis=1)
temp1 = np.concatenate((temp1, ad_val), axis=1)
valid_layer1 = np.concatenate((temp1, rd_val), axis=1)

RBM_layer1 = {'train':train_layer1,'valid':valid_layer1}
np.save(data_path+'RBM_layer1.npy', RBM_layer1) 
#%% Comparison between Gaussian and Bernoulli RBM
############# loading Bernoulli RBM
#bbrbm = BBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.05, momentum=0.9, use_tqdm=True)
#save_path='../models/'
#save_name='fa_model'
#bbrbm.load_weights(filename = save_path, name = save_name)
#errs_bbrbm = bbrbm.get_err(X_val_original)
#recon_val = bbrbm.reconstruct(X_val_original)
#error_bbrbm2 = np.square(X_val_original-recon_val).mean()

############## demean the data and loading Gaussian RBM
#X_s_mean = X_data.mean(axis=0)
#X_data = data_normalization(X_data)
#X_val = X_data[-n_val:]
#save_path='../models/GBRBM/rbm_fa/'
#save_name='fa_model'
#gbrbm = GBRBM(n_visible=X_data.shape[1], n_hidden=1000, learning_rate=0.01, momentum=0.9, use_tqdm=True)
#gbrbm.load_weights(filename = save_path, name = save_name)
#recon_val = gbrbm.reconstruct(X_val)
#transform_val = gbrbm.transform(X_val)
#recon_val += X_s_mean
#error_gaussian = np.square(X_val_original-recon_val).mean()

#%% Testing reconstruction image from Minst
#IMAGE = 10
#image = mnist_images[IMAGE]
#image_rec = gbrbm.reconstruct(image.reshape(1,-1))
#show_digit((image-0.1545)/0.33)
#show_digit(image_rec)
##%% transform the data into smaller dimensional image
#IMAGE = 1
#image = mnist_images[IMAGE]
#aaa=gbrbm.transform(image.reshape(1,-1))
