import pytest
import deepinv as dinv

from deepinv_bench import run_benchmark

TEST_BENCH = "cbsd500_gaussian_denoising"


def test_run_benchmark():
    """Make sure that the run_benchmark comand is properly running"""
    my_solver = dinv.models.DnCNN()
    results = run_benchmark(my_solver, TEST_BENCH, debug=True)
    assert isinstance(results, dict), "results should be a dict"
    for col in ["PSNR", "NIQE"]:
        assert col in results, f"{col} not found in results"
        assert f"{col}_std" in results, f"{col}_std not found in results"


def test_run_invalid_benchmark():
    """Make sure that the run_benchmark comand is properly running"""

    bench = "invalid_bench"
    err_msg = f"Could not find benchmark: {bench}"
    with pytest.raises(ValueError, match=err_msg):
        run_benchmark(None, "invalid_bench")


def test_run_invalid_method():
    """Make sure that the run_benchmark comand is properly running"""

    err_msg = "Model should be an instance of deepinv.models.Reconstructor"
    with pytest.raises(ValueError, match=err_msg):
        run_benchmark(None, TEST_BENCH)
