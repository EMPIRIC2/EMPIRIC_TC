### Here we define our UNet: (link paper)

from keras import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate, LeakyReLU,  Cropping2D

l1 = 0.00

def upsample_block(filters, up_sample_size, kernel_size, batch_norm=True, dropout=False):

    ## Upsample using nearest neighbors -> conv2d -> batchnorm -> dropout -> Relu

    result = Sequential()

    result.add(UpSampling2D(size = up_sample_size, interpolation="bilinear"))
    result.add(Conv2D(filters = filters, kernel_size=kernel_size, padding='same', kernel_initializer='HeNormal'))

    if batch_norm:
        result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))
    
    result.add(Conv2D(filters = filters, kernel_size=kernel_size, padding='same', kernel_initializer='HeNormal'))

    if batch_norm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def downsample_block(filters, size, batch_norm=True, dropout=False):

    # Downsample using conv2d -> Batchnorm -> Max Pool -> leakyReLU

    result = Sequential()
    result.add(
        Conv2D(filters, size, padding='same', kernel_initializer='HeNormal', activation=LeakyReLU()))

    if batch_norm:
        result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(
        Conv2D(filters, size, padding='same', kernel_initializer='HeNormal', activation=LeakyReLU()))

    if batch_norm:
        result.add(BatchNormalization())
 
    result.add(MaxPooling2D((2,2), padding='same'))
    
    result.add(LeakyReLU())
    return result

def bottleneck():
    result = Sequential()

    result.add(Dense(256*2, activation=LeakyReLU()))

    return result

def UNet(
        genesis_size,
        output_size,
        dropout,
        batch_norm,
        kernel_size,
        down_filters,
        up_filters
):

    assert len(down_filters) == len(up_filters)

    genesis = Input(shape=genesis_size)
    skips = []

    x = Conv2D(64, kernel_size, activation=LeakyReLU(), padding='same', kernel_initializer='HeNormal')(genesis)

    for i, filters in enumerate(down_filters):

        skips.append(x)
        x = downsample_block(filters, kernel_size, batch_norm=batch_norm, dropout=dropout)(x)

    x = Conv2D(down_filters[-1]*2, kernel_size, activation=LeakyReLU(), padding='same', kernel_initializer='HeNormal')(x)

    for i, filters in enumerate(up_filters):

        x = upsample_block(filters, (2,2), kernel_size, batch_norm=batch_norm, dropout=dropout)(x)
        if i < len(down_filters):
            skip = skips[len(down_filters) - i - 1]
            x = concatenate([x,skip])
            
    uncropped_output = Conv2D(output_size[-1], (1,1), kernel_initializer='HeNormal')(x)

    # crop the output
    output_size_diff_x = genesis_size[0] - output_size[0]
    output_size_diff_y = genesis_size[1] - output_size[1]

    output = Cropping2D(
            ((
                output_size_diff_x//2,
                output_size_diff_x//2
            ),
            (
                output_size_diff_y//2,
                output_size_diff_y//2
            )))(uncropped_output)

    model = Model(inputs=[genesis], outputs=[output])

    return model
