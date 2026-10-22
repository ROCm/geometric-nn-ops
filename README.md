# geometric-nn-ops

> Vendor-neutral GPU kernels for geometry-aware neural nets (triangle/folding ops)

## Overview

Vendor-neutral GPU kernels (forward + backward) for geometry-aware neural nets,
written in Triton that run on both AMD/ROCm and NVIDIA.

Kernels are namespaced by family:

- `geometric_nn_ops_torch.triangle` — protein-folding (triangle) ops.
  - `triangle_attention` (forward + backward)
  - `triangle_multiplicative_update` (Triton forward; pure-torch differentiable
    fallback)
  - `attention_pair_bias` (pure torch; no Triton kernel yet)

This repo ships two separately-installable packages:

- **`geometric-nn-ops-torch`**: the portable Triton kernels and their direct,
  vendor-neutral API. This is the recommended dependency.
- **`hipequivariance-ops-torch`**: an optional zero-code-change shim that
  implements the API of NVIDIA's `cuequivariance_ops_torch` under the same
  module name routes into these kernels. It translates any API differences. See
  [hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md).
  This is provided as a convenience for codebases already integrated with
  `cuequivariance`. For long term stability, we suggest using the direct
  dependency where possible.

## Status

Project Status: Active Development. Version 0.1.0; packages are not yet
published on PyPI.

| Op                               | Forward                                      | Backward                                    |
| -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| `triangle_attention`             | Triton                                       | Triton                                      |
| `triangle_multiplicative_update` | Triton; pure torch when gradients are needed | pure torch (native Triton backward pending) |
| `attention_pair_bias`            | pure torch (no Triton kernel yet)            | pure torch (no Triton kernel yet)           |

## Prerequisites

- Linux with Python 3.12 or later.
- PyTorch 2.13 or later and Triton 3.7 or later, built for your platform (ROCm or
  CUDA).
- Reference environment: ROCm 10.1, Ubuntu 24.04, Python 3.12, PyTorch 2.13
  (`rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0`).

## Installation

### Package Installation

Packages are not yet published on PyPI; build from source.

### Build From Source

Install PyTorch and Triton for your environment first, e.g. following
https://pytorch.org/get-started/locally/. Then:

```bash
git clone https://github.com/ROCm/geometric-nn-ops.git
cd geometric-nn-ops/
# --no-deps keeps pip from installing PyPI's torch/triton over your platform's builds
pip install --no-deps -e .
```

To also install the `cuequivariance_ops_torch` shim, see
[hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md#install).

## Usage

```python
from geometric_nn_ops_torch.triangle import triangle_attention

o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)
```

See the docstring for the appropriate tensor layouts and other options.

```python
from geometric_nn_ops_torch.triangle import triangle_multiplicative_update

z = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **weights)
```

`triangle_multiplicative_update` matches the cuequivariance signature
(concatenated `[a; b]` `p_in`/`g_in` projections). It runs the Triton fast path
for inference and auto-falls back to a pure-torch differentiable implementation
when gradients are needed or no GPU/Triton is available.

### fp32 matmul precision

For fp32 inputs, these ops read `torch.backends.cuda.matmul.fp32_precision`. In
the Triton kernels, `"tf32"` resolves in the same way as
`torch.set_float32_matmul_precision("high")` to whatever reduced-precision fp32
matmul the target has with at least TF32 precision — TF32 tensor cores on NVIDIA
Ampere and above, XF32 MFMA on AMD CDNA3, bf16 emulation elsewhere — and to full
fp32 where there is none.

## Documentation

Each op's docstring documents its tensor layouts and options. The
`cuequivariance_ops_torch` shim is documented in
[hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md).

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development workflow, coding standards, and how to submit pull requests. By
participating you agree to abide by the
[code of conduct](https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#code-of-conduct).

## Security

Please follow [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.
