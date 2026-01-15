# DeepInverse Benchmarks

This repository contains a set benchmarks for evaluating the performance
of different image reconstruction methods implemented in the DeepInverse library.

- **DeepInverse documentation**: https://deepinv.github.io
- **DeepInverse repository**: https://github.com/deepinv/deepinv

**Leaderboards** are automatically generated and can be found in the [DeepInverse benchmarks documentation](https://deepinv.github.io/deepinv/benchmarks.html).

Benchmark results are stored in a HuggingFace repository: https://huggingface.co/datasets/deepinv/benchmarks/tree/main

### Evaluating Your Reconstruction Methods

To evaluate your own reconstruction methods on these benchmarks,
install DeepInverse with the benchmarks extra:

```bash
pip install deepinv[benchmarks]
```

and then run on python:

```python
from deepinv.benchmarks import run_benchmark
my_solver = lambda y, physics: ...  # your solver here
results = run_benchmark(my_solver, "benchmark_name")
```

where  `benchmark_name` is the name of the benchmark and `my_solver` is your reconstruction method which receives `(y, physics)` where

- `y` is a `torch.Tensor` containing the measurements,
- `physics` is the forward operator, see more details in the [DeepInverse physics documentation](https://deepinv.github.io/deepinv/user_guide/physics/intro.html).

### Adding New Solvers

To add a new solver to an existing benchmark, open a new pull request on this repository, adding a new `your_solver_name.py` file
in the corresponding benchmark folder. Follow the structure of the existing solver files. 
The new solver will be automatically run once the pull request is merged, and the results will be added to the leaderboard.

### Adding New Benchmarks

To create a new benchmark, open a new pull request adding a new folder following
the structure given in the existing [benchmark_template](https://github.com/deepinv/benchmarks) folder.

A new benchmark requires: 

- A **dataset** from the [DeepInverse datasets](https://deepinv.github.io/deepinv/user_guide/training/datasets.html). 
- A **forward operator** from the [DeepInverse operators](https://deepinv.github.io/deepinv/user_guide/physics/physics.html).
- A set of **reconstruction methods** from the [DeepInverse reconstructors](https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html) and potentially other custom solvers.
- A set of **metrics** from the [DeepInverse metrics](https://deepinv.github.io/deepinv/user_guide/training/metric.html).

If you would like to propose a new dataset, metric or forward operator, please [open an issue](https://github.com/deepinv/benchmarks/issues/new/choose).