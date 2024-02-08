### Here we define our own modified UNet: (link paper)

from keras import Sequential, Model

from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate

def upsample_block(filters, size, dropout=False):

    ## Upsample using nearest neighbors -> conv2d -> batchnorm -> dropout -> Relu

    result = Sequential()
    result.add(UpSampling2D(size = size, interpolation="nearest"))
    result.add(Conv2D(filters = filters, kernel_size=(3,3), padding='same', use_bias=False))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(Activation.ReLU())

    return result

def downsample_block(filters, size, dropout=False):

    # Downsample using conv2d -> Batchnorm -> Max Pool -> leakyReLU

    result = Sequential()
    result.add(
        Conv2D(filters, size, strides=2, padding='same', use_bias=False, activation=Activation.LeakyReLU()))

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(
        Conv2D(filters, size, strides=2, padding='same', use_bias=False, activation=Activation.LeakyReLU()))

    result.add(BatchNormalization())

    result.add(MaxPooling2D((2,2), padding="same"))

    return result

def bottleneck(size):
    # we add a fully connected layer to try to capture non-local spatial interactions

    result = Sequential()
    result.add(Dense(size, activation=Activation.LeakyReLU()))

    return result


def UNet(input_size, output_channels = 1):

    input = Input(input_size)

    skips = []

    x = Conv2D(64, (3,3), activation=Activation.LeakyReLU)

    down_filters = [64, 128, 256]
    for filters in down_filters:

        x = downsample_block(filters, (3,3), dropout=True)(x)
        skips.append(x)

    x = bottleneck((input_size / 2**len(down_filters)) + (256,))

    ## outputs are twice as big as inputs
    up_filters = [256, 128, 64, 32]
    for i, filters in enumerate(up_filters):

        x = upsample_block(filters, (3,3), dropout=True)(x)
        if i < len(down_filters):
            x = concatenate([x, skips[len(down_filters) - i - 1]])

    output = Conv2D(output_channels, (1,1), activation=Activation.Softmax)(x)

    model = Model(inputs=[input], outputs=[output])

    return model