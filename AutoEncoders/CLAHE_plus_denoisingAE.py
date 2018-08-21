from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K
import cv2
import numpy as np
import os
from PIL import Image
import histogram_equalization as hist

"""
written by wooramkang 2018.08.20

referenced from lots of papers and gits
if you need those, i'll send you

"""



def main_color():
    # load the CIFAR10 data
    (x_train, _), (x_test, _) = cifar10.load_data()

    """
    for preprocessing,
    RGB to LAB
    
    for img in all of img
        DO CLAHE
    
    LAB to RGB 
    """

    x_train_prime = []
    for _img in x_train:
        x_train_prime.append(hist.preprocessing_hist(_img))
    x_train = np.array(x_train_prime)
    print(x_train.shape)

    x_test_prime = []
    for _img in x_test:
        x_test_prime.append(hist.preprocessing_hist(_img))
    x_test = np.array(x_test_prime)
    print(x_test.shape)
    """
    written by wooramkang 2018.08.21

    depending on CLAHE parameters,
    

    """
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]

    imgs_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), imgs_dir)
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    imgs = x_test[:100]

    i = 0
    for _img in imgs:
        i = i+1
        Image.fromarray(_img).save('saved_images/{0}_img_raw.png'.format(i))

    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    Image.fromarray(imgs).save('saved_images/sumof_img_raw.png')

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    print(x_train.shape)

    input_shape = (img_rows, img_cols, 3)

    batch_size = 32
    kernel_size = 3
    latent_dim = 256
    layer_filters = [64, 128, 256]

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)

    shape = K.int_shape(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')
    #loss_func =
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    outputs = Conv2DTranspose(filters=channels,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'AE_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    autoencoder.compile(loss='mse', optimizer='adam')

    callbacks = [lr_reducer, checkpoint]
    # .fit(data for train, data for groundtruth, validtation set, epochs, batchsize, ...)
    autoencoder.fit(x_train,
                    x_train,
                    validation_data=(x_test, x_test),
                    epochs=30,
                    batch_size=batch_size,
                    callbacks=callbacks)

    x_decoded = autoencoder.predict(x_test)

    imgs = x_decoded[:100]
    print(imgs.shape)
    imgs = (imgs * 255).astype(np.uint8)
    i = 0
    for _img in imgs:
        i = i + 1
        Image.fromarray(_img).save('saved_images/{0}_img_gen.png'.format(i))

    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    Image.fromarray(imgs).save('saved_images/sumof_img_gen.png')

if __name__ == "__main__":
    main_color()



