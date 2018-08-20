import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from keras.layers import ZeroPadding2D, Activation, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Deconv2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.local import LocallyConnected2D
import glob
import cv2


def img_path_to_encoding(image_path):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1)


def img_to_encoding(image):
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lumin, a, b = cv2.split(image)
    img = lumin
    print(img)
    output_arr = np.array([img])
    return output_arr


def convolution_AE():

    database = {}
    x_train = []
    encoding_dim = 64
    raw_dim = 512

    # load all the images of individuals to recognize into the database
    for file in glob.glob("../images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        print(file)
        temp = img_path_to_encoding(file)
        x_train.append(temp)
        database[identity] = temp

    print(database)
    x_list = [value for index, value in enumerate(database)]
    print(x_list)

    x_train= np.array(x_train)
    print(x_train)
    print(len(x_train))

    x_train = x_train.reshape(len(x_train),raw_dim, raw_dim)

    print(x_train.shape)
    """
        written by wooram 2018.08. 17

        1. hard to code convolutional AE from bottom

        am i missing something?
    """
    autoencoder = Sequential()

#ENCODER

    autoencoder.add(Dense(encoding_dim * 4, activation='relu', input_shape=x_train[1:].shape))
    autoencoder.add(Dense(encoding_dim * 2, activation='relu'))
    autoencoder.add(MaxPooling2D((2, 2),strides=2))
    autoencoder.add(MaxPooling2D((2, 2), strides=4))

    autoencoder.add(Dense(encoding_dim, activation='relu'))

#DECODER

    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dense(encoding_dim*2, activation='relu'))
    autoencoder.add(Dense(encoding_dim*4, activation='relu'))
    autoencoder.add(Dense(raw_dim, activation='relu'))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()


'''
if __name__ == "__main__":
    convolution_AE()
    autoencoder.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=x_train[1:].shape))
    autoencoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2)))
    autoencoder.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
'''