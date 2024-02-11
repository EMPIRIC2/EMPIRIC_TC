import keras
from dataset import get_dataset
from unet import UNet

genesis_size_default = (55, 105, 6)
movement_size_default = (11, 13)
output_size_default = (110, 210, 12)

def train_unet(data_folder, genesis_size=genesis_size_default, output_size=output_size_default):

    train_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size)
    print(train_data)
    test_data = get_dataset(data_folder, genesis_size=genesis_size, output_size=output_size, test=True)
    model = UNet(genesis_size_default)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error"),
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    )

    model.fit(
        x=train_data,
        epochs=1,
        verbose=2,
    )

    model.evaluate(
        x=test_data,
    )

    model.save('models/unet.keras')

keras.backend.clear_session()
train_unet('../Training Data Generation/Data/train/')

