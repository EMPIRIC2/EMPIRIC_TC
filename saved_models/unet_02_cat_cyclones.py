### Here we define our UNet: (link paper)

import tensorflow as tf
from tensorflow import keras
from MachineLearning.dataset import get_dataset
import time
import numpy as np
from keras import Sequential, Model, activations, regularizers

from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate, LeakyReLU, ReLU, Softmax, ZeroPadding2D, Cropping2D, Flatten, Reshape

l1 = 0.00

def upsample_block(filters, up_sample_size, kernel_size, dropout=False):

    ## Upsample using nearest neighbors -> conv2d -> batchnorm -> dropout -> Relu

    result = Sequential()

    result.add(UpSampling2D(size = up_sample_size, interpolation="nearest"))
    result.add(Conv2D(filters = filters, kernel_size=kernel_size, padding='same', kernel_initializer='HeNormal', use_bias=False, kernel_regularizer=regularizers.L1(l1=l1)))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))
    
    result.add(Conv2D(filters = filters, kernel_size=kernel_size, padding='same', kernel_initializer='HeNormal', use_bias=False, kernel_regularizer=regularizers.L1(l1=l1)))
   
    result.add(BatchNormalization())

    result.add(ReLU())

    return result

def downsample_block(filters, size, dropout=False):

    # Downsample using conv2d -> Batchnorm -> Max Pool -> leakyReLU

    result = Sequential()
    result.add(
        Conv2D(filters, size, padding='same', kernel_initializer='HeNormal', activation=LeakyReLU(), use_bias=False, kernel_regularizer=regularizers.L1(l1=l1)))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(
        Conv2D(filters, size, padding='same', kernel_initializer='HeNormal', activation=LeakyReLU(), use_bias=False, kernel_regularizer=regularizers.L1(l1=l1)))
    
    result.add(BatchNormalization())
 
    result.add(MaxPooling2D((2,2), padding='same'))

    return result

def bottleneck():
    result = Sequential()

    result.add(Dense(256*2, activation=LeakyReLU()))

    return result

def UNet02CatCyclones(genesis_size, output_channels = 1):

    genesis = Input(shape=genesis_size)
    #movement = Input(shape=movement_size)
    skips = []

    x = Conv2D(64, (3,3), activation=LeakyReLU(), padding='same', kernel_initializer='HeNormal', kernel_regularizer=regularizers.L1(l1=l1))(genesis)

    down_filters = [64*2, 128*2, 256*2]
    for i, filters in enumerate(down_filters):

        skips.append(x)
        x = downsample_block(filters, (3,3), dropout=False)(x)
    
    x = Conv2D(512, (3,3), activation=LeakyReLU(), padding='same', kernel_initializer='HeNormal')(x)
    
    ## outputs are twice as big as inputs
    up_filters = [256*2, 128*2, 64*2, 32*2]
    for i, filters in enumerate(up_filters):

        x = upsample_block(filters, (2,2), (3,3), dropout=False)(x)

        if i < len(down_filters):
            skip = skips[len(down_filters) - i - 1]
            b, h, w, c = skip.shape
            b, h1, w1, c = x.shape
            diffY = h1 - h
            diffX = w1 - w

            #padded_skip = ZeroPadding2D(padding=()(skip)
            
            cropped_x = Cropping2D(cropping=((diffY // 2, diffY - diffY//2), (diffX // 2, diffX - diffX // 2)))(x)

            x = concatenate([cropped_x,skip])
            
    output = Conv2D(output_channels, (1,1), kernel_initializer='HeNormal', activation=tf.keras.activations.elu, use_bias=False, kernel_regularizer=regularizers.L1(l1=l1))(x)
    
    model = Model(inputs=[genesis], outputs=[output])

    return model
