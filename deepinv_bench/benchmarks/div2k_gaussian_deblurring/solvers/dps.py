from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = "DPS"

    parameters = {
        "denoiser": ["DRUNet", "DiffUNet"],
        "max_iter": [1000],
    }

    test_config = {
        "max_iter": 1,
        "denoiser": "DnCNN",
    }

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        if self.denoiser == "DRUNet":
            denoiser = dinv.models.DRUNet(device=device)
        elif self.denoiser == "DiffUNet":
            denoiser = dinv.models.DiffUNet().to(device)
        elif self.denoiser == "DnCNN":
            denoiser = dinv.models.DnCNN().to(device)
        else:
            raise NotImplementedError

        self.model = dinv.sampling.DPS(
            model=denoiser, device=device, max_iter=self.max_iter
        )
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
