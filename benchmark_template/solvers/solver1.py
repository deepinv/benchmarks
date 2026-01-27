from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    name = 'solver_name'

    # add any hyper-parameters here
    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )

        # replace by your model, should take (y, physics) as input
        self.model = dinv.models.DnCNN(device=device)
        self.model.device = device

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
