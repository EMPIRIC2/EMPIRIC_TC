import keras
from dataset import get_dataset
from unet import UNet

genesis_size_default = (6, 55, 105)
movement_size_default = (11, 13)


def train_unet(data_folder, genesis_size=genesis_size_default, movement_size=movement_size_default):

    train_data = get_dataset(data_folder, genesis_size=genesis_size, movement_size=movement_size)
    test_data = get_dataset(data_folder, genesis_size=genesis_size, movement_size=movement_size, test=True)
    model = UNet((32,) + genesis_size_default)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MSE,
        metrics=[keras.metrics.MSE, keras.metrics.MeanAbsoluteError]
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