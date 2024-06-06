from saved_models.unet_02_cat_cyclones import UNet02CatCyclones
import pathlib

script_directory = pathlib.Path(__file__).parent.resolve()

print(script_directory)


class UNetCustom02CatCyclones:
    """
    Class to record the trained model (for when there are multiple models)
    """

    genesis_size = (55, 105, 1)
    model_path = script_directory / "unet_mean_1713754646.2664263.keras"

    @staticmethod
    def load_model():
        model = UNet02CatCyclones(UNetCustom02CatCyclones.genesis_size, 1)
        model.load_weights(UNetCustom02CatCyclones.model_path)
        return model
