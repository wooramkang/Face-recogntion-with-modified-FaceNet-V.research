import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import genfromtxt, regularizers
from keras import backend as K
from keras.layers import ZeroPadding2D, Activation, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshap
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense


def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)


def img_to_encoding(image, model):
    image = cv2.resize(image, (384, 384))
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    output_arr = np.array([img])
    return output_arr


def convolution_AE():

    database = {}
    encoding_dim = 96
    raw_dim = 384

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        database[identity] = img_path_to_encoding(file)

    x_train = [float(value)/256 for index, value in enumerate(database)]

    database_test = {}

    for file in glob.glob("test/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        database_test[identity] = img_path_to_encoding(file, FRmodel)

    x_test = [float(value)/256 for index, value in enumerate(database_test)]

    autoencoder = Sequential()
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(384, 384)))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(4, (2, 2), activation='relu', padding='same'))
    autoencoder.summary()


if __name__ == "__main__":
    convolution_AE()