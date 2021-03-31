import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tifffile as tif
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
X_TRAIN_PATH = 'final_dataset/train/cells/'
Y_TRAIN_PATH='final_dataset/train/cell_masks/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

train_ids=sorted(os.listdir(X_TRAIN_PATH))
#print("train_ids",train_ids)
print("TRAINING DATA VERIFIED")

def equal_ratio_rgb(gray):
  h=gray.shape[0]
  w=gray.shape[1]
  c=gray.shape[2]
  maxi=max(h,w)
  actual=np.zeros((maxi,maxi,c))
  for i in range(c):
    if(h<w):
      extra=[[0]*w for i in range(w-h)]
      actual[:,:,i]=np.vstack((gray[:,:,i],extra))
    else:
      extra=[[0]*(h-w) for i in range(h)]
      actual[:,:,i]=np.hstack((gray[:,:,i],extra))
  return actual

def equal_ratio(gray):
  h=gray.shape[0]
  w=gray.shape[1]
  if (h<w):
    extra=[[0]*w for i in range(w-h)]
    actual=np.vstack((gray,extra))
  else:
    extra=[[0]*(h-w) for i in range(h)]
    actual=np.hstack((gray,extra))
  return actual

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  path = X_TRAIN_PATH + id_
  img = cv2.imread(path)[:,:,:IMG_CHANNELS]
  img=equal_ratio_rgb(img)
  img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
  X_train[n] = img
  mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
  path=Y_TRAIN_PATH+id_
  mask_ = cv2.imread(path,0)
  mask_=equal_ratio(mask_)
  mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
  mask_= np.expand_dims(mask_,axis=-1)
  mask = np.maximum(mask, mask_)
  Y_train[n] = mask
  
print("PREPROCESSING DONE!!")
print("sum of X_train",np.sum(X_train))

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
#import tensorlayer as tl


def Unet(img_size):
    inputs = Input((img_size, img_size, 3))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

from tensorflow.keras.losses import binary_crossentropy
model = Unet(img_size=128)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('train_unet.h5', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[tf.keras.metrics.Precision()])
model.fit(X_train,Y_train,validation_split=0.1,batch_size=5, epochs=10,callbacks=[earlystopper, checkpointer])
print("TRAINING DONE!!")

model = load_model('train_unet.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
print("PREDICTION DONE!!!")


ix=172
print(model.evaluate(tf.expand_dims(X_train[ix],axis=0),tf.expand_dims(Y_train[ix],axis=0)))
cv2.imwrite('training_img2.png', X_train[ix])
print("Y_train",np.sum( Y_train[ix]))
cv2.imwrite('mask_img2.png', Y_train[ix]*255)
print("preds_train_t",np.sum(preds_train_t[ix]))
cv2.imwrite('pred_img2.png',preds_train_t[ix]*255)
print("OUPUTS SAVED")