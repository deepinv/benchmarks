from benchopt import BaseDataset
from benchopt.config import get_data_path

import deepinv as dinv
from torchvision import transforms
from deepinv.datasets import DIV2K
from deepinv.physics import (
    Blur,
    GaussianNoise,
)


class Dataset(BaseDataset):

    name = "DIV2K"
    parameters = {
        'sigma': [0.1],
        'img_size': [256],
        'debug': [False],
    }

    requirements = ["datasets"]

    def get_data(self):

        root = get_data_path("DIV2K")
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        dataset = DIV2K(
            root, mode="val", download=True, transform=transform
        )
        if self.debug:
            dataset = dinv.torch.utils.data.Subset(dataset, [0, 1, 2])

        return dict(
            dataset=dataset,
            physics=Blur(
                filter=dinv.physics.blur.gaussian_blur(2),
                noise_model=GaussianNoise(0.05)
            ),
        )
