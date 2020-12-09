import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
def load_data(labels):
	x=[]
	y=[]
	for i in labels:
		images=os.listdir("train/"+i)
		for j in images:
			img=imread("train/"+i+"/"+j)
			img=resize(img,(64,64))
			x.append(img)
			y.append(labels.index(i))
	x=np.array(x)
	y=np.array(y)
	return x,y

labels=os.listdir("train")
print(labels)

# load data
x,y = load_data(labels)

#pre-process

def preprocess(x):
        x=x.astype('float32')
        x=x/255.
        return x

x=preprocess(x)

# one  hot encoder

def onehotencode(y):
        y=np_utils.to_categorical(y)
        classes=y.shape[1]
        return y,classes

y,classes=onehotencode(y)

#splitting the dataset
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2, random_state=7)

epochs=10

#model

def model_cats_dogs(classes,epoch):
        model=Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        model.compile(optimizer=sgd(lr=0.01, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.summary())
        return model

model=model_cats_dogs(classes,epoch)

#fitting the model
history=model.fit(xtrain,ytrain, validation_data=(xtest,ytest),epochs=epoch, batch_size=32)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

test_path="test/"
test_ids=os.listdir(test_path)
test=[]
for i in test_ids:
        img=imread(test_path+"/"+i)
        img=resize(img,(64,64))
        test.append(img)
predict=model.predict(test)
print(predict)
eval=model.evaluate(xtest,ytest)
print("accuracy",eval[1])



