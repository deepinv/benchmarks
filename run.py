from pathlib import Path

import torch
import benchopt
import pandas as pd
import deepinv as dinv


BENCHMARK_ROOT = Path(__file__).parent


def run_benchmark(
        benchmark_name: str,
        model: dinv.models.Reconstructor | torch.nn.Module,
        model_name: str = None,
        debug: bool = False,
):
    r"""
    Run a benchmark on a given model.


    :param str benchmark_name: Name of the benchmark to run.
    :param dinv.models.Reconstructor | torch.nn.Module model:
    :param str model_name: optional name of the model to display in the results.
    :param bool debug: Whether to run the benchmark in debug mode.
    :return: dict with benchmark results, including metrics and runtime.
    """
    model_name = model_name or model.__class__.__name__

    benchmark = benchopt.benchmark.Benchmark(BENCHMARK_ROOT / benchmark_name)
    objectives = benchmark.check_objective_filters([])
    datasets = benchmark.check_dataset_patterns([f"*[debug={debug}]"])

    # By default
    try:
        solvers = benchmark.check_solver_patterns(["runner-solver"])
        if len(solvers) == 1:
            raise ValueError
    except Exception:
        from base.runner_solver import Solver
        solvers = [(Solver, {})]
    solver = solvers[0][0]
    solver.model = model
    solver.name = model_name

    # Run the benchmark
    exit_code, fname = benchopt.runner._run_benchmark(
        benchmark=benchmark,
        solvers=solvers,
        datasets=datasets,
        objectives=objectives,
        plot_result=False,
        pdb=True
    )
    if exit_code != 0:
        raise RuntimeError("Benchmark run failed.")
    results = pd.read_parquet(fname)
    col = [c for c in results.columns if "objective_" in c]
    results = results[["solver_name"] + col]
    results = results.rename(columns=lambda x: x.replace("objective_", ""))

    return results


if __name__ == "__main__":
    solver = dinv.models.RAM()
    results = run_benchmark(
        "div2k_gaussian_deblurring", solver,
        model_name="RAM", debug=True
    )
    print(results)
