
from MachineLearning.dataset import get_dataset
from MachineLearning.ConvProbPredictor.conv_prob_predictor import *
import time
import tensorflow as tf
import keras

def sparse_categorical_focal_cross_entropy(y_true, y_pred):
    max_storms = 8
    ### convert y_real to one hot vectors and then compute the cross-entropy loss

    y_true = tf.cast(y_true, tf.int8)
    y_one_hot = tf.one_hot(y_true, max_storms)

    return keras.losses.CategoricalFocalCrossentropy()(y_pred, y_one_hot)



def train(data_folder):
    train_time = time.time()
    train_data = get_dataset(data_folder, data_version=2)
    #test_data = get_dataset(data_folder, data_version=1, dataset="test")
    validation_data = get_dataset(data_folder, data_version=2, dataset="validation")


    # update this!
    genesis_shape = (55, 105, 1)
    movement_shape = (13,)
    num_outputs = 542

    save_path = "models/site_prob_negbin_{}.weights.h5".format(str(train_time))
    early_stopping = keras.callbacks.EarlyStopping()
    checkpoint = keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True, mode='min', verbose=1)
    model = conv_prob_predictor(genesis_shape, movement_shape, num_outputs)

    # TODO: add CPRS metric
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=NegLogLikNegBinomial
    )

    model.fit(train_data,
              epochs=20,
              validation_data=validation_data,
              verbose=2,
              callbacks=[early_stopping, checkpoint]
             )

    #model.evaluate(
       # test_data,
      #  verbose=2
    #)

    #model.save_weights("models/site_prob_{}.weights.h5".format(str(time.time())))
    #model.save('models/site_prob_{}.keras'.format(str(time.time())))
