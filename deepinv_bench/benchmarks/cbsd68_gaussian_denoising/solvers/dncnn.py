from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = "DnCNN"

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        class RescaledDenoiser(dinv.models.Denoiser):
            def __init__(self, model, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.base_model = model

            def forward(self, x, sigma):
                # DnCNN is trained for sigma=2.0/255, so we rescale the input and output
                sigma_ref = 2.0 / 255
                factor = sigma_ref / sigma
                return self.base_model(x * factor) / factor

        rescaled_denoiser = RescaledDenoiser(dinv.models.DnCNN())
        self.model = dinv.models.ArtifactRemoval(rescaled_denoiser, device=device)
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
