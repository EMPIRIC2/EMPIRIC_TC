from torch.utils import data
from MachineLearning.dataset import hdf5_generator_UNets
import numpy as np
import glob
import os

class IterDataset(data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator

    def __getitem__(self):
        x, y = next(iter(self))
        return {'x': x.astype(np.float32), 'y': y.astype(np.float32)}

def get_pytorch_dataloader(folder_path, batch_size, dataset="train"):
    file_paths = glob.glob(os.path.join(folder_path, "*.hdf5"))

    generator = hdf5_generator_UNets(file_paths, dataset=dataset)

    db = IterDataset(generator)

    dataloader = data.DataLoader(
        db,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return dataloader