from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = "DiffPIR"

    parameters = {"denoiser": ["DRUNet", "DiffUNet"], "zeta": [0.95]}

    def set_objective(self, train_dataset=None, physics=None):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

        if self.denoiser == "DRUNet":
            denoiser = dinv.models.DRUNet(device=device)
        elif self.denoiser == "DiffUNet":
            denoiser = dinv.models.DiffUNet().to(device)
        else:
            raise NotImplementedError

        self.model = dinv.sampling.DiffPIR(
            model=denoiser, zeta=self.zeta, data_fidelity=dinv.optim.L2(), device=device
        )
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
