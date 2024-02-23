import tensorflow as tf
from tensorflow import keras
from dataset import get_dataset

genesis_size = (55, 105, 6)
movement_size = (11, 13)
output_size = (110, 210, 12)

def evaluations(model_path, data_folder):
    test_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size, test=True)
    
    model = tf.keras.models.load_model(model_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error"),
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError(), keras.metrics.MeanAbsolutePercentageError()]
    )

    model.evaluate(
        x=test_data,
        verbose=2
    )

evaluations("models/unet.keras", "Data/run1/")
