from benchopt import BaseSolver

import torch
import torch.nn as nn
import deepinv as dinv


class Solver(BaseSolver):
    name = "TVDenoiser"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )

        _tv = dinv.models.TVDenoiser()
        _device = device

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.tv = _tv
                self.device = _device

            def forward(self, y, physics, **kwargs):
                # Use the noise sigma as the TV regularization weight
                gamma = physics.noise_model.sigma
                return self.tv(y.to(self.device), gamma)

        self.model = _Wrapper()

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)

