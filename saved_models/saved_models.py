from saved_models.UNet_02_cat_cyclones import UNet02CatCyclones
from saved_models import ddpm_unet
from saved_models import ddpm_unet_no_attention
from saved_models.SongUNet import get_network
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class UNetCustom02CatCyclones:
    """
    Class to record the trained model (for when there are multiple models)
    """

    MODEL_CONFIG = {
        "genesis_size": (112, 224, 1),
        "output_size": (110, 210, 1),
        "kernel_size": (5,5),
        "dropout": False,
        "batch_norm": True,
        "down_filters": [8, 16, 32],
        "up_filters": [32, 16, 8]
    }

    model_path = os.path.join(__location__, "UNet-Custom_1718056468.1028786.keras")

    @staticmethod
    def load_model():
        model = UNet02CatCyclones(**UNetCustom02CatCyclones.MODEL_CONFIG)
        model.load_weights(UNetCustom02CatCyclones.model_path)
        return model

class SongUNet:
    MODEL_CONFIG = {
        "input_size": (112, 224, 1),
        "output_size": (110, 210, 1),
        "widths": [8, 32, 64],
        "block_depth": 2
    }
    
    model_path = os.path.join(__location__,"SongUNet_1717729429.7140238.keras")
    
    @staticmethod
    def load_model():
        model = get_network(**SongUNet.MODEL_CONFIG)
        model.load_weights(SongUNet.model_path)
        return model

    
class DDPMUNetNoAttention02CatCyclones:
    
    MODEL_CONFIG = {
        "img_size": (112, 224, 1),
        "output_size": (110, 210, 1),
        "has_attention": [False, False, False],
        "interpolation": "bilinear",
        "widths": [8, 16, 32],
        "include_temb": False
    }

    model_path = os.path.join(__location__, "DDPM-Unet_1718057631.7278447.keras")
    
    @staticmethod
    def load_model():
        model = ddpm_unet_no_attention.build_model(**DDPMUNetNoAttention02CatCyclones.MODEL_CONFIG)
        model.load_weights(DDPMUNetNoAttention02CatCyclones.model_path)
        return model

class DDPMUNet02CatCyclones:
    
    MODEL_CONFIG = {
        "img_size": (112, 224, 1),
        "output_size": (110, 210, 1),
        "has_attention": [False, False, True],
        "interpolation": "bilinear",
        "widths": [16, 32, 64],
        "include_temb": False
    }
    
    model_path = os.path.join(__location__, "DDPM-Unet_1718141443.4486032.keras")
    
    @staticmethod
    def load_model():
        model = ddpm_unet.build_model(**DDPMUNet02CatCyclones.MODEL_CONFIG)
        model.load_weights(DDPMUNet02CatCyclones.model_path)
        return model