from MachineLearning.UNet.unet import UNet

"""
Object keeps track of the models we want to save and evaluate.
Weights and model imports need to be updated if the class locations or weight files change. E.g. possibly on different machines.
"""
models_info = [{
    "Name": "Custom UNet",
    "Output": "Mean 0-2 Category TCs over 10 years",
    "weights": "...",
    "model": UNet,
    "params": ((55, 105, 1), (11, 13), 1)
}]
