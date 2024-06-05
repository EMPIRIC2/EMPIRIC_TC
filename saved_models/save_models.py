from unet_02_cat_cyclones import UNet02CatCyclones

class UNetCustom02CatCyclones:
    """
    Class to record the trained model (for when there are multiple models)
    """

    genesis_size = (55, 105, 1)
    model_path = "./unet_mean_1713754646.2664263.keras"

    @staticmethod
    def load_model():
        model = UNet02CatCyclones(UNetCustom02CatCyclones.genesis_size, 1)
        model.load_weights(UNetCustom02CatCyclones.model_path)
        return model
