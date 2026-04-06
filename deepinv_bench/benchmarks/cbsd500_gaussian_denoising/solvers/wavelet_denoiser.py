from benchopt import BaseSolver

import torch
import torch.nn as nn
import deepinv as dinv


class Solver(BaseSolver):
    name = "WaveletDenoiser"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )

        _wavelet = dinv.models.WaveletDenoiser(device=device)
        _device = device

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.wavelet = _wavelet
                self.device = _device

            def forward(self, y, physics, **kwargs):
                # Use the noise sigma as the wavelet threshold
                sigma = physics.noise_model.sigma
                return self.wavelet(y.to(self.device), ths=sigma)

        self.model = _Wrapper()

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)

