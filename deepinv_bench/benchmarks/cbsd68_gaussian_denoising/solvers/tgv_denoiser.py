from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = "TGVDenoiser"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        class MyDenoiser(dinv.models.Reconstructor):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, y, physics, **kwargs):
                # finetuned ths value for best perf
                return self.model(y, ths=7 * physics.noise_model.sigma)

        self.model = MyDenoiser(model=dinv.models.TGVDenoiser())
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
