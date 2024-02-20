import tensorflow_probability as tfp
import tensorflow as tf
from keras import Sequential, Model, optimizers, activations
from keras.layers import Layer, Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, concatenate, Flatten, LeakyReLU, ReLU
from dataset import get_dataset

tfd = tfp.distributions


class TPNormalMultivariateLayer(Layer):
    def __init__(self, num_prob_outputs=2):
        super().__init__()
        self.dense = Dense(num_prob_outputs + num_prob_outputs * (num_prob_outputs - 1) / 2)
        
        self.num_prob_outputs = num_prob_outputs

        self.locs_fn = Dense(self.num_prob_outputs, ReLU())

        # we don't actually use it here because of tensorflow bug, but keep it defined for when it needs to be used later
        self.dist = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=tf.squeeze(t[..., :1]), scale_tril=t[..., 1:]))

    def call(self, inputs):

        x = self.dense(inputs)

        triang = tfp.math.fill_triangular(x)

        pos_diag = activations.softplus(tf.linalg.diag_part(triang)) + 1e-3

        C = tf.linalg.set_diag(triang, pos_diag)

        covariance = tf.matmul(C, tf.linalg.matrix_transpose(C))
        loc = tf.expand_dims(self.locs_fn(x), axis=-1)

        x = concatenate([loc, covariance], axis=-1)

        return x

def NegLogLik(n_outputs):
    # a custom negative log likelihood loss
    # we use the network to get the params for a distribution
    # and then calculate that neg log likelihood using that dist.
    # this is because of a bug in Tensorflow/Keras where Tensor.log_prob is not found

    def loss(y_true, y_pred):

        distribution = lambda t: tfd.MultivariateNormalTriL(loc=tf.squeeze(t[..., :1]), scale_tril=t[..., 1:])

        y_params = y_pred
        y_dist = distribution(y_params)

        negloglik = lambda p_y, y: -p_y.log_prob(y)

        return negloglik(y_dist, y_true)

    return loss

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

    conv.add(Flatten())

    x = conv(inputs)
    x = Dense(1024, LeakyReLU())(x)
    outputs = TPNormalMultivariateLayer(num_outputs)(x)

    model = Model(inputs=inputs,outputs=outputs)

    return model


def train(data_folder):

    train_data = get_dataset(data_folder, data_version=1)
    test_data = get_dataset(data_folder, data_version=1, dataset="test")
    validation_data = get_dataset(data_folder, data_version=1, dataset="validation")


    # update this!
    input_shape = (55, 105, 1)
    num_outputs = 542

    model = build_conv_prob_predictor(input_shape, num_outputs)

    # TODO: add CPRS metric
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=NegLogLik(num_outputs)
    )

    model.fit(train_data, validation_data=validation_data, verbose=2)

    model.evaluate(
        test_data,
        verbose=2
    )

train("/nesi/nobackup/uoa03669/storm_data/v2")