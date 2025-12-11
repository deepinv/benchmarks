import abc
import deepinv as dinv
from typing import Any


class Benchmark(abc.ABC):
    r"""
    Abstract base class for benchmarks

    All of the benchmarks should inherit this class and implement the `run` method.
    """

    @abc.abstractmethod
    def run(
        self,
        model: dinv.models.Denoiser | dinv.models.Reconstructor,
        *,
        device: torch.device | str = torch.device("cpu")
    ) -> Any:
        """Run the benchmark on the given model

        :param dinv.models.Denoiser | dinv.models.Reconstructor model: The model to benchmark
        :param torch.device | str device: The device to run the benchmark on (default: `"cpu"`)
        :return: (`Any`) The result of the benchmark
        """
        pass
