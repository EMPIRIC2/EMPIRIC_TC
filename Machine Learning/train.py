import tensorflow as tf
from tensorflow import keras
from dataset import get_dataset
from unet import UNet
import time

genesis_size_default = (55, 105, 6)
movement_size_default = (11, 13)
output_size_default = (110, 210, 6)

def train_unet(data_folder, epochs=50,genesis_size=genesis_size_default, output_size=output_size_default):
 
    train_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size)
    
    test_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size, test=True)
    model = UNet(genesis_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error"),
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        x=train_data.skip(2),
        epochs=epochs,
        verbose=2,
        validation_data=train_data.take(2)
    )

    model.evaluate(
        x=test_data,
    )

    model.save('models/unet_{}.keras'.format(time.time()))

keras.backend.clear_session()
train_unet('Data/run1/')

