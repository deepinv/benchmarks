from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = 'DPIR'

    parameters = {
        'sigma': [0.1],
    }

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.model = dinv.optim.DPIR(
            sigma=self.sigma, device=device
        )
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
