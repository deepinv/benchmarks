from benchopt import BaseSolver

import torch
import torch.nn as nn
import deepinv as dinv


class Solver(BaseSolver):
    name = "EPLLDenoiser"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )

        _epll = dinv.models.EPLLDenoiser(device=device)
        _device = device

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.epll = _epll
                self.device = _device

            def forward(self, y, physics, **kwargs):
                sigma = physics.noise_model.sigma
                return self.epll(y.to(self.device), sigma)

        self.model = _Wrapper()

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)

