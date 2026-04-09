from pathlib import Path

import torch
import benchopt
import pandas as pd
import deepinv as dinv

BENCHMARK_ROOT = Path(__file__).parent / "benchmarks"


def run_benchmark(
    model: dinv.models.Reconstructor | torch.nn.Module,
    benchmark_name: str,
    model_name: str = None,
    debug: bool = False,
):
    r"""
    Run a benchmark on a given model.


    :param str benchmark_name: Name of the benchmark to run. The benchmark
        name should match a folder in the benchmarks directory.
    :param dinv.models.Reconstructor | torch.nn.Module model:
    :param str model_name: optional name of the model
        to display in the results.
    :param bool debug: Whether to run the benchmark in debug mode.
    :return: dict with benchmark results, including metrics and runtime.
    """
    model_name = model_name or str(model.__class__.__name__)

    try:
        benchmark = benchopt.benchmark.Benchmark(BENCHMARK_ROOT / benchmark_name, no_cache=True)
    except Exception:
        all_benchmarks = "\n-".join(
            [
                p.name
                for p in BENCHMARK_ROOT.iterdir()
                if p.is_dir()
                and not p.name.startswith(".")
                and "template" not in p.name
            ]
        )
        raise ValueError(
            f"Could not find benchmark: {benchmark_name}.\n"
            f"Available benchmarks are:\n-{all_benchmarks}\n\n"
            "If the requested benchmark is not present, consider updating "
            "deepinv_bench by running \n\n"
            "pip install --upgrade --force-reinstall --no-deps "
            "git+https://github.com/deepinv/benchmarks.git"
        )

    if not isinstance(model, (dinv.models.Reconstructor, torch.nn.Module)):
        raise ValueError(
            "Model should be an instance of "
            "deepinv.models.Reconstructor or torch.nn.Module"
        )

    objectives = benchmark.check_objective_filters([])
    datasets = benchmark.check_dataset_patterns([f"*[debug={debug}]"])

    # By default
    try:
        solvers = benchmark.check_solver_patterns(["runner-solver"])
        if len(solvers) == 1:
            raise ValueError
    except Exception:
        from deepinv_bench.base.runner_solver import Solver

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
        pdb=False,
    )
    if exit_code != 0:
        raise RuntimeError("Benchmark run failed.")
    results = pd.read_parquet(fname)
    col = [c for c in results.columns if "objective_" in c]
    results = results[["solver_name"] + col]
    results = results.rename(columns=lambda x: x.replace("objective_", ""))
    # to dict
    results = results.to_dict("records")[0]
    return results


if __name__ == "__main__":
    solver = dinv.models.RAM()
    results = run_benchmark(
        solver, "div2k_gaussian_deblurring", model_name="RAM", debug=True
    )
    print(results)
