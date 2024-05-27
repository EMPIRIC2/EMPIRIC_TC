from SavedModelCode.unet_02_cat_cyclones import UNet02CatCyclones
from enum import Enum
class UNetCustom02CatCyclones:
    """
    Class to record to the trained model (for when there are multiple models)
    """

    genesis_size = (55, 105, 1)
    model_path = "../models/unet_mean_1713754646.2664263.keras"

    @staticmethod
    def load_model():
        model = UNet02CatCyclones(UNetCustom02CatCyclones.genesis_size, 1)
        model.load_weights(UNetCustom02CatCyclones.model_path)
        return model

class Models(Enum):
    UNetCustom02CatCyclones = 1
def predict(input):

    model = UNetCustom02CatCyclones.load_model()

    return model(input)
