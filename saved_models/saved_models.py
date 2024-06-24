from saved_models import ddpm_unet

import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class DDPMUNet:
    
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
        model = ddpm_unet.build_model(**DDPMUNet.MODEL_CONFIG)
        model.load_weights(DDPMUNet.model_path)
        return model