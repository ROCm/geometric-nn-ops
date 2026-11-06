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
  module name and routes calls into these kernels. It translates any API differences. See
  [hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md).
  This is provided as a convenience for codebases already integrated with
  `cuequivariance`. For long term stability, we suggest using the direct
  dependency where possible.

## Status

Project Status: Active Development. Version 1.0.0; packages are not yet
published on PyPI.

| Op                               | Forward                                      | Backward                                    |
| -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| `triangle_attention`             | Triton                                       | Triton                                      |
| `triangle_multiplicative_update` | Triton; pure torch when gradients are needed | pure torch (native Triton backward pending) |
| `attention_pair_bias`            | pure torch (no Triton kernel yet)            | pure torch (no Triton kernel yet)           |

## Prerequisites

- Linux with Python 3.12.
- PyTorch 2.13.0 and Triton 3.7.1, built for your platform (ROCm or CUDA).

Supported ROCm configuration:

| ROCm       | Ubuntu | Python | PyTorch             | AMD Instinct GPU                                                 |
| ---------- | ------ | ------ | ------------------- | ---------------------------------------------------------------- |
| 10.0.0, 10.1.0 | 24.04  | 3.12   | 2.13.0 (ROCm build) | gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI350X, MI355X) |

The recommended environment is the ROCm 10.1.0 PyTorch Docker image,
`rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0`, which
provides ROCm, PyTorch, and Triton.

## Installation

Packages are not yet published on PyPI; build them from source. Both packages
are pure Python, so building them does not need the ROCm compiler toolchain.

> [!IMPORTANT]
> Always pass `--no-deps` when building or installing these packages. pip
> checks the `torch` and `triton` requirements by package name, and ROCm Triton
> is also shipped as `triton-rocm` or `pytorch-triton-rocm`; when a requirement
> looks unmet, pip installs PyPI's CUDA builds over the ROCm ones.

Clone the repository on the host and start the container with the clone
mounted:

```bash
git clone https://github.com/ROCm/geometric-nn-ops.git
cd geometric-nn-ops/
docker run -it --rm \
  --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$PWD":/geometric-nn-ops -w /geometric-nn-ops \
  --name=geometric_nn_ops \
  rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0
```

Inside the container, build and install the wheel:

```bash
pip wheel . --no-deps -w ~/dist/  # fetches the hatchling build backend
pip install --no-deps ~/dist/geometric_nn_ops_torch-*.whl
```

For development, or in any environment that already has ROCm (or CUDA) builds
of PyTorch and Triton, install in editable mode instead:

```bash
pip install --no-deps -e .
```

To also install the `cuequivariance_ops_torch` shim, see
[hipequivariance_ops_torch/README.md](./hipequivariance_ops_torch/README.md#install).

### Verify the Installation

Run these outside the clone, so Python imports the installed package rather
than the source tree in the current directory:

```bash
cd ~
python3 -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} hip={torch.version.hip}")'
python3 -c 'import geometric_nn_ops_torch; print(geometric_nn_ops_torch.__version__)'
```

The first should print `cuda_available=True` and the HIP version of your
PyTorch build; the second prints the installed package version.

To run the test suite on the GPU, from the repository root
(`/geometric-nn-ops` in the container):

```bash
pip install "pytest==9.0.3"  # not the [test] extra, which would pull deps over --no-deps
python3 -m pytest -vrxs tests
```

> [!NOTE]
> Two `fp16` tests are expected to fail and appear as `2 xfailed` in the pytest
> summary. They cover fully masked rows, where softmax is undefined, and do not
> affect normal use.

If you installed the shim, add `hipequivariance_ops_torch/tests` to test it too.
Tests marked `large` need datacenter-class GPU memory and are deselected by
default; add `-m ""` to include them. Tests against cuequivariance skip unless
`cuequivariance-torch` is installed.

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

## Maintainers

| Role           | Contact                                                    |
| -------------- | ---------------------------------------------------------- |
| Technical Lead | @GMNGeoffrey, @singagan                                    |
| Maintainer     | @demandal25, @zhihuidu-amd, @rtmadduri, @mukh1l, @yaoliu13 |

## License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.
