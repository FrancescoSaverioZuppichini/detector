import torch
from torch.utils import benchmark


def profile_on_cuda(fn, min_run_time=30):
    # warmup
    for _ in range(4):
        fn()
    print("[INFO] profiling...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt=f"fn()",
        globals={"fn": fn},
        label="profile",
        sub_label="",
        description="",
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2**20
    print(
        f"[INFO] mean={res.mean * 1000}ms, median={res.median * 1000}ms, memory={memory} MB"
    )
    return res.mean * 1000, res.median * 1000, res.number_per_run, memory
