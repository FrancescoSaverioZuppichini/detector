from utils import profile_on_cuda
from torch import nn
import torch
import numpy as np
from time import perf_counter

# @profile_on_cuda

from scipy.optimize import linear_sum_assignment

import torch

import torch

# Load the shared library
torch.ops.load_library("my_kernel.so")

# Wrap the CUDA kernel in a Python function
def hungarian(x):
    # assert x.is_cuda and x.dtype == torch.float32, "Input tensor must be a float32 CUDA tensor"
    # N = x.numel()
    # threads_per_block = 256
    # blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Call the custom CUDA kernel
    torch.ops.my_namespace.Hungarian_Algorithm(x)


@torch.no_grad()
def sinkhorn_knopp(cost_matrix, epsilon=0.01, max_iter=1000, tau=1e-3):
    n, m = cost_matrix.shape
    assert n == m, "Cost matrix must be square"

    K = torch.exp(-cost_matrix / epsilon)
    u = torch.ones(n, device=cost_matrix.device) / n
    v = torch.ones(m, device=cost_matrix.device) / m

    u_prev = torch.empty(n, device=cost_matrix.device)
    v_prev = torch.empty(m, device=cost_matrix.device)

    for _ in range(max_iter):
        u_prev.copy_(u)
        v_prev.copy_(v)

        u.reciprocal_()
        torch.matmul(K, v, out=u)
        u.reciprocal_()

        v.reciprocal_()
        torch.matmul(K.t(), u, out=v)
        v.reciprocal_()

        if (
            torch.max(torch.abs(u - u_prev)) < tau
            and torch.max(torch.abs(v - v_prev)) < tau
        ):
            break

    P = torch.diag(u) @ K @ torch.diag(v)

    row_assignments = torch.argmax(P, dim=1)
    col_assignments = torch.argmax(P, dim=0)

    return row_assignments, col_assignments


def optimal_transport(M, lam=0.01, epsilon=1e-8, max_iter=1000):
    n, m = M.shape
    # Kinit = torch.exp(- M * lam)
    K = torch.exp(-M / epsilon)
    # somehow faster
    u = torch.ones(n, device=M.device) / n
    v = torch.ones(m, device=M.device) / m
    i = 0
    u_prev = torch.empty(n, device=M.device)
    v_prev = v.clone()

    while (torch.abs(v - v_prev).sum() > epsilon) or i > max_iter:
        u_prev.copy_(u)
        v_prev.copy_(v)
        # changing order affects convergence a little bit
        u.reciprocal_()
        torch.matmul(K, v, out=u)
        u.reciprocal_()

        v.reciprocal_()
        torch.matmul(K.t(), u, out=v)
        v.reciprocal_()
        i += 1

    print(i)
    P = torch.diag(u) @ K @ torch.diag(v)

    row_assignments = torch.argmax(P, dim=1)
    col_assignments = torch.argmax(P, dim=0)

    return row_assignments, col_assignments


# start = perf_counter()
C = np.random.rand(8500, 100)
# linear_sum_assignment(C)

# print(f"Took {perf_counter() - start:.2f}")

# with torch.autocast("cuda", torch.float16):
C = torch.from_numpy(C).float()
start = perf_counter()
hungarian(C)
# optimal_transport(C)
print(f"Took {perf_counter() - start:.2f}")


# C = torch.from_numpy(C).float().cuda()

# profile_on_cuda(lambda: sinkhorn_knopp(C))
