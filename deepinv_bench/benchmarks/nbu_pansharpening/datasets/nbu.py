from benchopt import BaseDataset
from benchopt.config import get_data_path

import deepinv as dinv
from torchvision import transforms


class Dataset(BaseDataset):
    name = "NBU"

    parameters = {
        "physics": ["Pansharpening"],
        "noise": ["ZeroNoise"],
        "img_size": [256],
        "factor": [4],
        "return_pan": [False],
        "satellite": ["gaofen-1"], # 4 channels dataset
    }

    test_parameters = {"debug": [True]}

    def get_data(self):
        root = get_data_path("NBUDataset")

        transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

        dataset = dinv.datasets.NBUDataset(
            root_dir="NBU",
            satellite=self.satellite,
            return_pan=self.return_pan,
            download=True,
            transform_ms=transform,
        )

        physics = dinv.physics.Downsampling(
            filter='bilinear',
            factor=4,
            img_size=(4, self.img_size, self.img_size),
        )

        return dict(
            dataset=dataset,
            physics=physics,
        )