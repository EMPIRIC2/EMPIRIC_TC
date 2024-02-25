
from MachineLearning.dataset import get_dataset
from MachineLearning.ConvProbPredictor.conv_prob_predictor import *
import time

def train(data_folder):

    train_data = get_dataset(data_folder, data_version=2)
    #test_data = get_dataset(data_folder, data_version=1, dataset="test")
    validation_data = get_dataset(data_folder, data_version=2, dataset="validation")


    # update this!
    genesis_shape = (55, 105, 1)
    movement_shape = (13,)
    num_outputs = 542

    callback = keras.callbacks.EarlyStopping()

    model = conv_prob_predictor(genesis_shape, movement_shape, num_outputs )


    # TODO: add CPRS metric
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=NegLogLik
    )

    model.fit(train_data,
              epochs=20,
              validation_data=validation_data,
              verbose=2,
              callbacks=[callback]
             )

    #model.evaluate(
       # test_data,
      #  verbose=2
    #)

    model.save_weights("models/site_prob_{}.weights.h5".format(str(time.time())))
    #model.save('models/site_prob_{}.keras'.format(str(time.time())))
