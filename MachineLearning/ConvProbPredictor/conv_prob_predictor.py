import tensorflow_probability as tfp
import tensorflow as tf
from keras import Sequential, Model, optimizers, activations
from keras.layers import Reshape, Layer, Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, concatenate, Flatten, LeakyReLU, ReLU
import keras

tfd = tfp.distributions
tf.keras.utils.get_custom_objects().clear()
@keras.saving.register_keras_serializable(package="ProbabilisticLayers")
class TPNormalMultivariateLayer(Layer):
    def __init__(self, num_prob_outputs=2):
        super().__init__()

        self.num_prob_outputs = num_prob_outputs
        # we don't actually use it here because of tensorflow bug, but keep it defined for when it needs to be used later
        #self.dist = tfp.layers.DistributionLambda(
            #lambda t: tfd.MultivariateNormalDiag(loc=tf.squeeze(t[..., :1]), scale_tril=t[..., 1:]))

    def build(self, input_shape):
        self.dense = Dense(self.num_prob_outputs + self.num_prob_outputs * (self.num_prob_outputs - 1) / 2, kernel_regularizer='l2')

        self.locs_fn = Dense(self.num_prob_outputs, ReLU(), kernel_regularizer='l2')

    def get_config(self):
        return {'num_prob_outputs': self.num_prob_outputs}

    def call(self, inputs):

        x = self.dense(inputs)

        triang = tfp.math.fill_triangular(x)

        pos_diag = activations.softplus(tf.linalg.diag_part(triang)) + 1e-3

        C = tf.linalg.set_diag(triang, pos_diag)

        covariance = tf.matmul(C, tf.linalg.matrix_transpose(C))
        loc = tf.expand_dims(self.locs_fn(x), axis=-1)

        x = concatenate([loc, covariance], axis=-1)

        return x

@keras.saving.register_keras_serializable(package="ProbabilityLayers", name="NegLogLikDiscrete")
def NegLogLikDiscrete(y_true, y_pred):

    distribution = lambda t: tfd.FiniteDiscrete(logits=t)
    y_dist = distribution(y_pred)
    negloglik = lambda p_y, y: -p_y.log_prob(y)

    return negloglik(y_dist, y_true)

@keras.saving.register_keras_serializable(package="ProbabilityLayers", name="NegLogLikPoisson")
def NegLogLikPoisson(y_true, y_pred):
    # a custom negative log likelihood loss
    # we use the network to get the params for a distribution
    # and then calculate that neg log likelihood using that dist.
    # this is because of a bug in Tensorflow/Keras where Tensor.log_prob is not found


    distribution = lambda t: tfd.Poisson(total_count=t[:])
    y_params = y_pred
    y_dist = distribution(y_params)

    negloglik = lambda p_y, y: -p_y.log_prob(y)

    return negloglik(y_dist, y_true)

@keras.saving.register_keras_serializable(package="ProbabilityLayers", name="NegativeLogLikBinomial")
def NegLogLikNegBinomial(y_true, y_pred):
    num_sites = 542

    distribution = lambda t: tfd.NegativeBinomial(total_count=t[:num_sites], probs=t[num_sites:])

    y_dist = distribution(y_pred)

    negloglik = lambda p_y, y: -p_y.log_prob(y)

    return negloglik(y_dist, y_true)


def conv_prob_predictor(genesis_shape, movement_shape, num_outputs, max_storms = 8):
    genesis_matrix = Input(genesis_shape)
    movement_coefficients = Input(movement_shape)
    conv = Sequential()

    n_filters = [64, 128, 256]
    for filters in n_filters:
        conv.add(Conv2D(filters, (3, 3), padding='same', activation=LeakyReLU()))
        conv.add(BatchNormalization())

        conv.add(Conv2D(filters, (3, 3), padding='same', activation=LeakyReLU()))
        conv.add(BatchNormalization())

        conv.add(MaxPooling2D((2, 2), padding='same'))

    conv.add(Flatten())

    x = conv(genesis_matrix)
    x = concatenate([x, movement_coefficients])

    x = Dense(1000, activation=LeakyReLU(), kernel_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_outputs * max_storms, activation=LeakyReLU(), kernel_regularizer='l2')(x)

    output = Reshape((num_outputs, max_storms))(x)

    model = Model(inputs=[genesis_matrix, movement_coefficients], outputs=output)

    return model
