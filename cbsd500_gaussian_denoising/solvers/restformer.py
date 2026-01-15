from benchopt import BaseSolver

import deepinv as dinv


class Solver(BaseSolver):
    name = 'Restormer'

    parameters = {}

    def set_objective(self, train_dataset=None, physics=None):
        self.model = dinv.models.ArtifactRemoval(
            dinv.models.Restormer()
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(model=self.model)
