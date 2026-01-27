from benchopt import BaseDataset
from benchopt.config import get_data_path

import deepinv as dinv
from torchvision import transforms


class Dataset(BaseDataset):

    # configure the dataset name here
    name = "dataset_name"

    # define the parameters to be used in the benchmark
    # IMPORTANT: the names of physics and noise should match
    # the ones defined in deepinv.physics with exact same spelling
    parameters = {
        'physics': ['physics_name'],
        'noise': ['noise_name'],
        'img_size': [256],
        # add any other parameter you might need
    }

    def get_data(self):
        root = get_data_path("Set14_HR")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        # load the dataset
        dataset = dinv.datasets.Set14HR(
            root, download=True, transform=transform
        )

        # define the physics according to the parameters
        physics = dinv.physics.Denoising(
            noise_model=dinv.physics.GaussianNoise()
        )

        return dict(
            dataset=dataset,
            physics=physics,
        )
