from benchopt import BaseDataset
from benchopt.config import get_data_path

from torchvision import transforms
import deepinv as dinv
from deepinv.datasets import CBSD68
from deepinv.physics import (
    Denoising,
    GaussianNoise,
)


class Dataset(BaseDataset):

    name = "CBSD68"

    parameters = {
        'img_size': [256],
        'sigma': [0.1],
        'debug': [False],
    }

    requirements = ["datasets"]

    def get_data(self):

        root = get_data_path("CBSD68")
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        dataset = CBSD68(
            root, download=True, transform=transform
        )

        if self.debug:
            dataset = dinv.torch.utils.data.Subset(dataset, [0, 1, 2])

        return dict(
            dataset=dataset,
            physics=Denoising(GaussianNoise(sigma=self.sigma)),
        )
