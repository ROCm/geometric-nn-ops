# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test utils for torch precision setting."""

import contextlib

import torch


@contextlib.contextmanager
def torch_matmul_precision(precision: str):
    """Set torch's fp32 matmul precision for the duration.

    Swaps a process global, torch having no scoped API, so this is not thread
    safe (hence test only). Both the forward and the backward must be called
    inside the context.
    """
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    matmul.fp32_precision = precision
    try:
        yield
    finally:
        matmul.fp32_precision = previous
