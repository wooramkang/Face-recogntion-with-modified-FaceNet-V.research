from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Average, Conv2DTranspose
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10, mnist
from keras.utils import plot_model
from keras import backend as K
import cv2
import numpy as np
import os
from PIL import Image
import histogram_equalization as hist


def main_model():
    # load the CIFAR10 data
    (x_train, _), (x_test, _) = cifar10.load_data()
    #(x_train, _), (x_test, _) = mnist.load_data()

    x_train_prime = []
    for _img in x_train:
        #_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        _img = cv2.resize(_img, (64, 64))
        _img = hist.preprocessing_hist(_img)
        x_train_prime.append(_img)
    x_train = np.array(x_train_prime)
    print(x_train.shape)

    x_test_prime = []
    for _img in x_test:
        #_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        _img = cv2.resize(_img, (64, 64))
        _img = hist.preprocessing_hist(_img)
        x_test_prime.append(_img)

    x_test = np.array(x_test_prime)
    print(x_test.shape)

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
    '''
    written by wooramkang 2018.08.27
    from line 0 to here
    copied-and-pasted from my AutoEncoders codes
    
    from next-line to end
    i wrote 
    '''
    input_shape = (img_rows, img_cols, 3)

    batch_size = 32
    layer_filters = [(32,1),(32,3),(32,5),(32,7)]

    inputs = Input(shape=input_shape, name='model_input')
    x = inputs
    output_layer = []

    x = Conv2D(filters=64,
              kernel_size=(3,3),

              strides=1,
              activation='relu',
              padding='same')(x)

    for filters, kernel_size in layer_filters:
        output_layer.append(Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   activation='relu',
                                   padding='same')(x))

    avg_output = Average()(output_layer)

    out = Conv2D(3, (3,3), activation='relu', padding='same', name ='finaloutput')(avg_output)

    model = Model(inputs, out, name='model')
    model.summary()

    SupResolution = Model(inputs, model(inputs),name='super-resolution')
    SupResolution.summary()

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'resolution_model.{epoch:03d}.h5'
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

    SupResolution.compile(loss='mse', optimizer='adam')

    #callbacks = [lr_reducer, checkpoint]
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    callbacks = [early, lr_reducer, checkpoint]

    SupResolution.fit(x_train,
                      x_train,
                      validation_data=(x_test, x_test),
                      epochs=30,
                      batch_size=batch_size,
                      callbacks=callbacks)

    x_decoded = SupResolution.predict(x_test)

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
    '''
    written by wooramkang 2018.08.28
    if images's size are quite small, then the effect of this network would be not that much as much as you expected
      
    '''
if __name__ == "__main__":
    main_model()



