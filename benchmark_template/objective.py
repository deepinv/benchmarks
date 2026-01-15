from benchopt import BaseObjective

import deepinv as dinv
from torch.utils.data import DataLoader


class Objective(BaseObjective):

    # modify name of the benchmark
    name = "benchmark_name"

    url = "https://github.com/deep-inverse/benchmarks"

    requirements = ["deepinv"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.8"

    sampling_strategy = 'run_once'

    def set_data(self, dataset, physics):
        self.dataset = dataset
        self.physics = physics

    def evaluate_result(self, model):
        device = getattr(model, 'device', None)
        self.physics = self.physics.to(device)

        # change metrics if needed
        metrics = [
            dinv.loss.PSNR(),
            dinv.loss.NIQE(device=device)
        ]

        results = dinv.test(
            model,
            DataLoader(self.dataset),
            self.physics,
            online_measurements=True,
            device=device,
            metrics=metrics,
            compare_no_learning=False
        )

        return results

    def get_one_result(self):
        return dict(model=lambda x: x)

    def get_objective(self):
        return dict(
            train_dataset=self.dataset,
            physics=self.physics,
        )
