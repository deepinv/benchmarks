from benchopt import BaseObjective

import time
import deepinv as dinv
from torch.utils.data import DataLoader


class Objective(BaseObjective):

    name = "DIV2K Gaussian Deblurring"

    url = "https://github.com/deep-inverse/benchmarks"

    requirements = ["deepinv"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.8"

    # Deactivate multiple runs for each solver
    sampling_strategy = "run_once"

    def set_data(self, dataset, physics):
        self.dataset = dataset
        self.physics = physics

    def evaluate_result(self, model):
        device = getattr(model, 'device', None)
        self.physics = self.physics.to(device)

        metrics = [
            dinv.loss.PSNR(),
            dinv.loss.NIQE(device=device)
        ]
        t_start = time.perf_counter()
        results = dinv.test(
            model,
            DataLoader(self.dataset),
            self.physics,
            online_measurements=True,
            device=device,
            metrics=metrics,
            compare_no_learning=False
        )
        results['runtime'] = time.perf_counter() - t_start

        return results

    def get_one_result(self):

        class DummyModel:
            def eval(self): pass

            def __call__(self, x, physics=None):
                return physics.A_adjoint(x)

        return dict(model=DummyModel())

    def get_objective(self):
        return dict(
            train_dataset=self.dataset,
            physics=self.physics,
        )
