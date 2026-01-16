import deepinv as dinv
import torch
import benchopt

def run_benchmark(model : dinv.models.Reconstructor | torch.nn.Module, benchmark_name : str):
    r"""
    Run a benchmark on a given model.


    :param dinv.models.Reconstructor | torch.nn.Module model:
    :param str benchmark_name: Name of the benchmark to run.
    :return: dict with benchmark results, including metrics and runtime.
    """

    # TODO: how can we do this with benchopt?

    return results


if __name__ == "__main__":
    solver = dinv.models.RAM()
    results = run_benchmark(solver, "div2k_gaussian_deblurring")
    print(results)