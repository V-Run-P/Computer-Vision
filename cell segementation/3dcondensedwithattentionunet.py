from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input,Add, MaxPooling3D, Dropout, concatenate, Conv3DTranspose
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
from tensorflow.keras.activations import sigmoid
import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.io import imread

import keras.backend as K

def f1_metric(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val


def attention(tensor, att_tensor,filters,kernel_size=[1,1,1]):
  g1=Conv3D(filters,kernel_size=kernel_size,strides=1)(tensor)
  x1=Conv3D(filters,kernel_size=kernel_size,strides=1)(att_tensor)
  net=Add()([g1,x1])
  relu_net=tf.nn.relu(net)  
  conv_net=Conv3D(1, kernel_size=kernel_size,strides=1)(relu_net)
  sig_net= sigmoid(conv_net)
  net=sig_net*att_tensor
  return net


def cunet3d(img_size):
  inputs=Input((16,img_size,img_size,1))

  c11= Conv3D(32,(5,5,5),activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
  c11= Dropout(0.1)(c11)
  c12= Conv3D(32,(5,5,5),activation='elu', kernel_initializer='he_normal',padding='same')(c11)
  c12= Dropout(0.1)(c12)
  c13= Conv3D(32,(5,5,5),activation='elu',kernel_initializer='he_normal', padding='same')(c12)
  p1=MaxPooling3D((4,4,4))(c13)
  
  #step2
  c21= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(p1)
  c21= Dropout(0.1)(c21)
  c22= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(c21)
  c22= Dropout(0.1)(c22)
  c23= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(c22)
  p2=MaxPooling3D((4,4,4))(c23)

  #step3
  c31= Conv3D(512,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(p2)
  c31= Dropout(0.1)(c31)
  c32= Conv3D(512,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(c31)
  c32= Dropout(0.1)(c32)
  c33= Conv3D(512,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(c32)

  #upsampling
  #step 1
  u21= Conv3DTranspose(128,(5,5,5),strides=(4,4,4),activation='elu',kernel_initializer='he_normal',padding='same')(c33)
  u21= attention(u21,c23,128)
  u21=concatenate([u21,c23])
  u22= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u21)
  u22= Dropout(0.1)(u22)
  u23= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u22)
  u23= Dropout(0.1)(u23)
  u24= Conv3D(128,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u23)

  #step2
  u11= Conv3DTranspose(32,(5,5,5),strides=(4,4,4),activation='elu',kernel_initializer='he_normal',padding='same')(u24)
  u11= attention(u11,c13,32)
  u11=concatenate([u11,c13])
  u12= Conv3D(32,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u11)
  u12= Dropout(0.1)(u12)
  u13= Conv3D(32,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u12)
  u13= Dropout(0.1)(u13)
  u14= Conv3D(32,(5,5,5),activation='elu',kernel_initializer='he_normal',padding='same')(u13)

  #last step
  output=Conv3D(1,(1,1,1), activation='sigmoid')(u14)
  
  model=Model(inputs=[inputs],outputs=[output])
  model.summary()
  return model

#training the model
image_path = "overlap_dataset/train/resampled_cells/"
mask_path="overlap_dataset/train/resampled_cell_masks/"

train_ids=sorted(os.listdir(image_path))

img_height=512
img_width=512
img_depth=16

x_train= np.zeros((len(train_ids),img_depth,img_height,img_width,1),dtype=np.uint8)
y_train= np.zeros((len(train_ids),img_depth,img_height,img_width,1),dtype=np.bool)
for n,id in tqdm(enumerate(train_ids),total=len(train_ids)):
        path=os.path.join(image_path,id)
        img=imread(path)
        x_train[n]=img
        paths=os.path.join(mask_path,id)
        msk=imread(paths)
        y_train[n]=msk
print(x_train.shape,y_train.shape)

from tensorflow.keras.losses import binary_crossentropy
model=cunet3d(img_size=512)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('3D_cond_att_overlap_dataset_09_05_2021.h5', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),f1_metric])
model.fit(x_train,y_train,validation_split=0.2, batch_size=1, epochs=100,callbacks=[earlystopper, checkpointer])
print("TRAINING DONE!!")
