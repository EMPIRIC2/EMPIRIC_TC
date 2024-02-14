import tensorflow_probability as tfp
import tensorflow as tf
from keras import Sequential, Model, activations, optimizers
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate, LeakyReLU, ReLU, Softmax, ZeroPadding2D, Cropping2D
tfd = tfp.distributions

def build_conv_prob_predictor(input_shape, num_outputs):

    inputs = Input(input_shape)

    conv = Sequential()

    n_filters = [64,128,256,512]
    for filters in n_filters:
        conv.add(Conv2D(filters, (3,3), padding='same', activation=LeakyReLU()))
        conv.add(BatchNormalization())

        conv.add(Conv2D(filters, (3,3), padding='same', activation=LeakyReLU()))
        conv.add(BatchNormalization())
        conv.add(Dropout(0.5))

        conv.add(MaxPooling2D((2,2), padding='same'))

    x = conv(inputs)
    x = Dense(1024, LeakyReLU())(x)

    x = Dense(num_outputs * 2)(x)

    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[...:num_outputs], scale=1e-3 + tf.math.softplus(0.05*t[..., num_outputs:])))(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def train(data_folder):



    negloglik = lambda y, p_y: -p_y.log_prob(y)

    # update this!
    input_shape = ()
    num_outputs = 10

    model = build_conv_prob_predictor(input_shape, num_outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.05),
        loss=negloglik
    )

    model.fit(train_data, validation_data=validation_data)
