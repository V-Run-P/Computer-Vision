from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input,Add, MaxPooling3D, Dropout, concatenate, Conv3DTranspose
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
from tensorflow.keras.activations import sigmoid
import time
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

def attention_unet(img_size):
    inputs = Input((16, img_size, img_size, 1))

    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = attention(u6, c4, 128)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u6 = attention(u7, c3, 64)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = attention(u8, c2, 32)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = attention(u9, c1, 16)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv3D(1, (1, 1, 1),activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
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
model=attention_unet(img_size=512)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('3D_attention_overlap_dataset_09_05_20213datt.h5', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),f1_metric])
model.fit(x_train,y_train,validation_split=0.2, batch_size=1, epochs=100,callbacks=[earlystopper, checkpointer])
print("TRAINING DONE!!")

