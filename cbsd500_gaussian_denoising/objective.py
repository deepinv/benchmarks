from benchopt import BaseObjective

import time
import deepinv as dinv
from torch.utils.data import DataLoader


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Denoising on CBSD68"

    url = "https://github.com/deep-inverse/benchmarks"

    requirements = ["deepinv"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.7"

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
        return dict(model=lambda x: x)

    def get_objective(self):
        return dict(
            train_dataset=self.dataset,
            physics=self.physics,
        )
