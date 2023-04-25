import pandas as pd
import typer
from pathlib import Path
from time import perf_counter
from functools import partial
from torch.utils import benchmark
import torch
from typing import Tuple
from pytorch_dataset import get_benchmark_func as pytorch_get_benchmark_func
from pytorch_transform_on_gpu_dataset import (
    get_benchmark_func as pytorch_pytorch_transform_on_gpu_dataset,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NAME_TO_BENCHMARK = {
    "pytorch-vanilla": pytorch_get_benchmark_func,
    "pytorch-gpu": pytorch_pytorch_transform_on_gpu_dataset,
}


def save_to_csv(**kwargs):
    df = pd.DataFrame.from_records([kwargs])
    file_path = RESULTS_DIR / "results.csv"
    df.to_csv(
        RESULTS_DIR / "results.csv",
        mode="a+",
        index=False,
        float_format="%.4f",
        header=not file_path.exists(),
    )


def with_perf_counter(benchmark_func, *args, **kwargs):
    # call it so it does its initialisation
    data_iterator_func = benchmark_func(*args, **kwargs)
    start = perf_counter()
    list(data_iterator_func())
    end = perf_counter()

    return {"total_time_ms": end - start}


def with_pytorch_benchmark(benchmark_func, *args, min_run_time=30, **kwargs):
    # call it so it does its initialisation
    data_iterator_func = benchmark_func(*args, **kwargs)
    # warmup
    for _ in range(2):
        list(data_iterator_func())

    def _wrapper():
        list(data_iterator_func())

    print("[INFO] profiling...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt=f"_wrapper()",
        globals={"_wrapper": _wrapper, "data_iterator_func": data_iterator_func},
        label="profile",
        sub_label="",
        description="",
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2**20
    memory_mb = f"Memory used: {memory} MB"
    print(f"[INFO] time={res} memory={memory_mb}")
    return {
        "mean": res.mean * 1000,
        "median": res.median * 1000,
        "number_per_run": res.number_per_run,
        "memory_mb": memory,
    }


def main(
    benchmark_name: str,
    dataset: Path,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    fp16: bool = False,
    compile: bool = False,
):
    params_dict = dict(
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        size=image_size,
        fp16=fp16,
        compile=compile,
    )

    benchmark_func = NAME_TO_BENCHMARK[benchmark_name]
    print(f"[INFO] Using {benchmark_name}")
    print(params_dict)
    res = with_pytorch_benchmark(benchmark_func, dataset, **params_dict)

    save_to_csv(benchmark_name=benchmark_name, **params_dict, **res)


if __name__ == "__main__":
    typer.run(main)
