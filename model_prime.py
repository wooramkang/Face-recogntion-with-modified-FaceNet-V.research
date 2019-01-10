import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU

_FLOATX = 'float32'

def variable(value, dtype=_FLOATX, name=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    _get_session().run(v.initializer)
    return v


def shape(x):
    return x.get_shape()


def square(x):
    return tf.square(x)


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def LRN2D(x):
    return tf.nn.lrn(x, alpha=1e-4, beta=0.75)


"""
    written by wooram 2018.08.14

    keras erased LRN2D before so there are just few options
    1. make same one
    2. find similar one


"""


def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer + '_conv' + num)(
        x)

    tensor = BatchNormalization(epsilon=0.00001, name=layer + '_bn' + num)(tensor)
    #tensor = LeakyReLU(0.2)(tensor)
    tensor = LeakyReLU(0.2)(tensor)

    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer + '_conv' + '2')(
        tensor)

    tensor = BatchNormalization( epsilon=0.00001, name=layer + '_bn' + '2')(tensor)
    #tensor = LeakyReLU(0.2)(tensor)
    tensor = LeakyReLU(0.2)(tensor)


    return tensor

def inception_block_1a(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name='inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(epsilon=0.00001, name='inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = LeakyReLU(0.2)(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = LeakyReLU(0.2)(X_3x3)
    X_5x5 = Conv2D(16, (1, 1), data_format='channels_first', name='inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = LeakyReLU(0.2)(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)
    X_5x5 = Conv2D(32, (5, 5), data_format='channels_first', name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = LeakyReLU(0.2)(X_5x5)

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization( epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)
    X_pool = LeakyReLU(0.2)(X_pool)
    X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)), data_format='channels_first')(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization( epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)
    X_1x1 = LeakyReLU(0.2)(X_1x1)

    # CONCAT
    inception = Concatenate( axis=channel_axis)([X_3x3, X_5x5, X_pool, X_1x1])

    return inception


def inception_block_1b(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name='inception_3b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = LeakyReLU(0.2)(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization( epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = LeakyReLU(0.2)(X_3x3)

    X_5x5 = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3b_5x5_conv1')(X)
    X_5x5 = BatchNormalization( epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = LeakyReLU(0.2)(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)
    X_5x5 = Conv2D(64, (5, 5), data_format='channels_first', name='inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization( epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = LeakyReLU(0.2)(X_5x5)

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization( epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)
    X_pool = LeakyReLU(0.2)(X_pool)
    X_pool = ZeroPadding2D(padding=(4, 4), data_format='channels_first')(X_pool)

    X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_1x1_conv')(X)
    X_1x1 = BatchNormalization( epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)
    X_1x1 = LeakyReLU(0.2)(X_1x1)

    inception = Concatenate( axis=channel_axis)( [X_3x3, X_5x5, X_pool, X_1x1])

    return inception


def inception_block_1c(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = conv2d_bn(X,
                        layer='inception_3c_3x3',
                        cv1_out=128,
                        cv1_filter=(1, 1),
                        cv2_out=256,
                        cv2_filter=(3, 3),
                        cv2_strides=(2, 2),
                        padding=(1, 1))

    X_5x5 = conv2d_bn(X,
                        layer='inception_3c_5x5',
                        cv1_out=32,
                        cv1_filter=(1, 1),
                        cv2_out=64,
                        cv2_filter=(5, 5),
                        cv2_strides=(2, 2),
                        padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(X_pool)

    inception = Concatenate( axis=channel_axis)([X_3x3, X_5x5, X_pool] )

    return inception


def inception_block_2a(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = conv2d_bn(X,
                        layer='inception_4a_3x3',
                        cv1_out=96,
                        cv1_filter=(1, 1),
                        cv2_out=192,
                        cv2_filter=(3, 3),
                        cv2_strides=(1, 1),
                        padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                        layer='inception_4a_5x5',
                        cv1_out=32,
                        cv1_filter=(1, 1),
                        cv2_out=64,
                        cv2_filter=(5, 5),
                        cv2_strides=(1, 1),
                        padding=(2, 2))

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                             layer='inception_4a_pool',
                             cv1_out=128,
                             cv1_filter=(1, 1),
                             padding=(2, 2))
    X_1x1 = conv2d_bn(X,
                            layer='inception_4a_1x1',
                            cv1_out=256,
                            cv1_filter=(1, 1))
    inception = Concatenate( axis=channel_axis)([X_3x3, X_5x5, X_pool, X_1x1] )

    return inception


def inception_block_2b(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = conv2d_bn(X,
                        layer='inception_4e_3x3',
                        cv1_out=160,
                        cv1_filter=(1, 1),
                        cv2_out=256,
                        cv2_filter=(3, 3),
                        cv2_strides=(2, 2),
                        padding=(1, 1))
    X_5x5 = conv2d_bn(X,
                        layer='inception_4e_5x5',
                        cv1_out=64,
                        cv1_filter=(1, 1),
                        cv2_out=128,
                        cv2_filter=(5, 5),
                        cv2_strides=(2, 2),
                        padding=(2, 2))

    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(X_pool)

    inception = Concatenate( axis=channel_axis)( [X_3x3, X_5x5, X_pool] )

    return inception


def inception_block_3a(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = conv2d_bn(X,
                        layer='inception_5a_3x3',
                        cv1_out=96,
                        cv1_filter=(1, 1),
                        cv2_out=384,
                        cv2_filter=(3, 3),
                        cv2_strides=(1, 1),
                        padding=(1, 1))
    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                         layer='inception_5a_pool',
                         cv1_out=96,
                         cv1_filter=(1, 1),
                         padding=(1, 1))
    X_1x1 = conv2d_bn(X,
                        layer='inception_5a_1x1',
                        cv1_out=256,
                        cv1_filter=(1, 1))

    inception = Concatenate( axis=channel_axis)([X_3x3, X_pool, X_1x1])

    return inception


def inception_block_3b(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X_3x3 = conv2d_bn(X,
                        layer='inception_5b_3x3',
                        cv1_out=96,
                        cv1_filter=(1, 1),
                        cv2_out=384,
                        cv2_filter=(3, 3),
                        cv2_strides=(1, 1),
                        padding=(1, 1))
    X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)
    X_pool = conv2d_bn(X_pool,
                             layer='inception_5b_pool',
                             cv1_out=96,
                             cv1_filter=(1, 1))
    X_pool = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_pool)

    X_1x1 = conv2d_bn(X,
                        layer='inception_5b_1x1',
                        cv1_out=256,
                        cv1_filter=(1, 1))
    inception = Concatenate(axis=channel_axis)([X_3x3, X_pool, X_1x1])

    return inception

def triplet_loss(y_true, y_pred, alpha=0.3):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

def FACE(input_shape):
    '''

    IMA_SIZE : 96 * 96
    F_SIZE : 128 ~ 196
    '''
    # Define the input as a tensor with shape input_shape

    dropout_rate = 0.4

    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)

    X = BatchNormalization(name='bn1')(X)
    X = LeakyReLU(0.2)(X)


    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    # Second Block
    X = Conv2D(64, (1, 1), strides=(1, 1), name='conv2')(X)

    X = BatchNormalization(epsilon=0.00001, name='bn2')(X)
    X = LeakyReLU(0.2)(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides=(1, 1), name='conv3')(X)

    X = BatchNormalization(epsilon=0.00001, name='bn3')(X)
    X = LeakyReLU(0.2)(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size=3, strides=2)(X)

    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)

    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)

    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)

    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(0.2)(X)

    X = Flatten()(X)

    #X = Dropout(dropout_rate)(X)
    X = Dense(256)(X)
    X = LeakyReLU(0.2)(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='FACE')
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    model.summary()

    return model