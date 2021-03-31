from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, Conv3DTranspose
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.io import imread

def Unet(img_size):
    inputs = Input((None, img_size, img_size, 1))

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
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

#training the model
images="final/preprocess_images/"
masks="final/preprocess_masks/"

train_ids=sorted(os.listdir(images))

img_height=256
img_width=256
img_depth=16

x_train=np.zeros((len(train_ids),img_depth,img_height,img_width,1),dtype=np.uint8)
y_train=np.zeros((len(train_ids),img_depth,img_height,img_width,1),dtype=np.bool)
for n,id in tqdm(enumerate(train_ids),total=len(train_ids)):
        path=os.path.join(images,id)
        img=imread(path)
        x_train[n]=img
        
        paths=os.path.join(masks,id)
        msk=imread(paths)
        y_train[n]=msk

        
print(x_train.shape,y_train.shape)
from tensorflow.keras.losses import binary_crossentropy
model = Unet(img_size=256)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('train_unet_with_epochs_100_batch_1.h5', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[tf.keras.metrics.Precision()])
model.fit(x_train,y_train,validation_split=0.1,batch_size=1, epochs=100,callbacks=[earlystopper, checkpointer])
print("TRAINING DONE!!")




