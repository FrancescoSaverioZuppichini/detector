import pandas as pd
import typer
from pathlib import Path
from pytorch_dataset import get_benchmark_func as pytorch_get_benchmark_func
from tensordict_dataset import get_benchmark_func as tensordict_get_benchmark_func
from time import perf_counter
from functools import partial

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NAME_TO_BENCHMARK = {
    "pytorch-vanilla": pytorch_get_benchmark_func,
    "tensordict": partial(
        tensordict_get_benchmark_func,
        dst=Path("/home/zuppif/Documents/neatly/detector/benchmarks/tmp"),
    ),
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


def main(
    benchmark_name: str,
    dataset: Path,
    num_iter: int = 5,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
):
    params_dict = dict(
        num_iter=num_iter,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    benchmark_func = NAME_TO_BENCHMARK[benchmark_name]
    print(f"Using {benchmark_name}")
    print(params_dict)
    res = with_perf_counter(benchmark_func, dataset, **params_dict)

    save_to_csv(benchmark_name=benchmark_name, **params_dict, **res)


if __name__ == "__main__":
    typer.run(main)
