#wget -O patch.exe "https://fileshare5010.depositfiles.com/auth-15079514569ca485e6a041223947bac1-27.34.22.182-266757333-180071176-guest/FS501-1/BO2-MP-ZM_LAN-patch_v3.exe" -c
#http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12070/l_mkl_2018.0.128.tgz
import numpy as np 
import dlib
import Image
import cv2 
# import tnfix	
import matplotlib.pyplot as plt 
import os 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.utils import shuffle 
# from keras.optimizers import adam
from sklearn.model_selection import GridSearchCV # new 
from sklearn.model_selection import train_test_split 
from keras.wrappers.scikit_learn import KerasClassifier # new 

data_img_list = np.load('numpy/m_and_f/data_img_list_150.npy')
dataset = np.array(data_img_list)
dataset = dataset.astype('float32')
dataset /=255
img_height = 150 
img_width = 150
#1012
dataset = dataset.reshape(dataset.shape[0],1,img_width, img_height)

num_classes = 2
number_of_images = dataset.shape[0]
labels = np.ones((number_of_images), dtype='int64')

labels[0:580] = 0 #male
labels[580:1012] = 1 #female

names = ['males', 'females']

Y = keras.utils.to_categorical(labels, num_classes)
x, y = shuffle(dataset, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0, random_state=2)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding="same", input_shape=(1,img_height,img_width), dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.250))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
# model.add(Convolution2D(256, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.250))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.250))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.250))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# sgd = keras.optimizers.SGD(
# 	lr=0.0005
# 	)

model.compile(
		loss='binary_crossentropy',
		optimizer= 'adam',
		metrics=['accuracy']
	)

history = model.fit(X_train, y_train, batch_size=5, nb_epoch=50, verbose=1, validation_data=(X_test,y_test))
model.save('kaggle_images/male_and_female.h5')

# scores = model.evaluate(X[test], Y[test], verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print(history.history.keys())
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

