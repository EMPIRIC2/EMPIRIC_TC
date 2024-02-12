### Here we define our own modified UNet: (link paper)

from keras import Sequential, Model, activations

from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate, LeakyReLU, ReLU, Softmax, ZeroPadding2D, Cropping2D

def upsample_block(filters, up_sample_size, kernel_size, dropout=False):

    ## Upsample using nearest neighbors -> conv2d -> batchnorm -> dropout -> Relu

    result = Sequential()

    result.add(UpSampling2D(size = up_sample_size, interpolation="nearest"))
    result.add(Conv2D(filters = filters, kernel_size=kernel_size, padding='same', use_bias=False))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result

def downsample_block(filters, size, dropout=False):

    # Downsample using conv2d -> Batchnorm -> Max Pool -> leakyReLU

    result = Sequential()
    result.add(
        Conv2D(filters, size, padding='same', use_bias=False, activation=LeakyReLU()))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(
        Conv2D(filters, size, padding='same', use_bias=False, activation=LeakyReLU()))

    result.add(BatchNormalization())

    result.add(MaxPooling2D((2,2), padding='same'))

    return result

def bottleneck():
    # we add a fully connected layer to try to capture non-local spatial interactions

    result = Sequential()
    result.add(Dense(256, activation=LeakyReLU()))

    return result


def UNet(input_size, output_channels = 12):

    input = Input(input_size)
    
    skips = []

    x = Conv2D(64, (3,3), activation=LeakyReLU(), padding='same')(input)
    
    down_filters = [64, 128, 256]
    for filters in down_filters:

        skips.append(x)
        x = downsample_block(filters, (3,3), dropout=True)(x)
        

    x = bottleneck()(x)

    ## outputs are twice as big as inputs
    up_filters = [256, 128, 64, 32]
    for i, filters in enumerate(up_filters):

        x = upsample_block(filters, (2,2), (3,3), dropout=True)(x)

        if i < len(down_filters):
            skip = skips[len(down_filters) - i - 1]
            b, h, w, c = skip.shape
            b, h1, w1, c = x.shape
            diffY = h1 - h
            diffX = w1 - w

            #padded_skip = ZeroPadding2D(padding=()(skip)
            cropped_x = Cropping2D(cropping=((diffY // 2, diffY - diffY//2), (diffX // 2, diffX - diffX // 2)))(x)
            x = concatenate([cropped_x,skip])

            

    output = Conv2D(output_channels, (1,1), activation=Softmax())(x)

    model = Model(inputs=[input], outputs=[output])

    return model

