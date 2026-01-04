from benchopt import BaseObjective

import time
from deepinv.loss import PSNR, NIQE


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
        results = []

        psnr = PSNR()
        niqe = NIQE()
        for i, X in enumerate(self.dataset):
            Y = self.physics.forward(X)
            t_start = time.perf_counter()
            X_pred = model(Y[None], self.physics)
            t_end = time.perf_counter()

            results += [
                dict(
                    NIQE=niqe(X_pred).item(),
                    PSNR=psnr(X_pred, X).item(),
                    runtime=t_end - t_start,
                    image_id=i
                )
            ]
            if i > 2:
                break

        return results

    def get_one_result(self):
        return dict(model=lambda x: x)

    def get_objective(self):
        return dict(
            train_dataset=self.dataset,
            physics=self.physics,
        )
