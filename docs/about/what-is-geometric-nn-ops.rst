.. meta::
  :description: What is geometric-nn-ops?
  :keywords: geometric-nn-ops, documentation, Triton, GPU, AMD, ROCm, overview, introduction, protein folding

.. _what-is-geometric-nn-ops:

********************************************************************
What is geometric-nn-ops?
********************************************************************

geometric-nn-ops is a library of vendor-neutral GPU kernels for geometry-aware
neural networks. Each operation provides a forward pass and a backward pass, and supports training and inference.

The kernels are written in `Triton <https://triton-lang.org/>`__ and called
through a PyTorch API. Because Triton compiles the same kernel source for
different GPU backends, geometric-nn-ops runs on AMD GPUs with ROCm and on
NVIDIA GPUs with CUDA from a single code base.

Kernels are grouped into families where each family is a submodule of
``geometric_nn_ops_torch``. The first family,
``geometric_nn_ops_torch.triangle``, contains the triangle operations used by
AlphaFold-style protein structure prediction models, such as AlphaFold and
OpenFold.

geometric-nn-ops kernels provide the following advantages:

- Triangle attention and the triangle multiplicative update are fused GPU
  kernels for pair-representation models. The fused kernels avoid materializing
  large intermediate tensors.

- A single Triton code base runs on AMD GPUs and on NVIDIA GPUs, so models
  don't need vendor-specific kernel paths.

- The ``hipequivariance-ops-torch`` compatibility package lets existing code
  written against NVIDIA's ``cuequivariance_ops_torch`` run on AMD GPUs without
  source changes.

Features
====================================================================

geometric-nn-ops provides the following features:

- **Triangle attention:** A Triton kernel for ``triangle_attention`` with
  forward and backward passes. It supports an optional pair bias and a boolean
  key mask, and accepts 4D, 5D, or 6D inputs. A variable-shape mode
  (``dynamic=True``) compiles one kernel for all sequence lengths, which avoids
  recompiling for every input size.

- **Triangle multiplicative update:** A Triton forward kernel for
  ``triangle_multiplicative_update`` in both the ``outgoing`` and ``incoming``
  directions. Its signature matches cuequivariance. When gradients are required,
  or when no GPU or Triton is available, it falls back automatically to a
  differentiable pure-PyTorch implementation.

- **Attention with pair bias:** ``attention_pair_bias()`` applies pair-bias
  layer normalization and projection, masked attention, gating, and the output
  projection, starting from pre-projected queries, keys, and values. This
  operation is currently implemented in pure PyTorch.

- **Precision control:** For ``fp32`` inputs, the Triton kernels read
  ``torch.backends.cuda.matmul.fp32_precision``. The ``"tf32"`` setting selects
  the fastest reduced-precision ``fp32`` matmul on the target with at least
  TF32 accuracy. For example, AMD CDNA3 GPUs use XF32 MFMA instructions, and
  other targets use ``bf16`` emulation. Targets without a reduced-precision
  ``fp32`` matmul use full ``fp32``.

- **Drop-in compatibility:** The optional ``hipequivariance-ops-torch`` package
  implements the ``cuequivariance_ops_torch`` module API and routes calls to the
  geometric-nn-ops kernels.

Use cases
====================================================================

geometric-nn-ops is designed for the following scenarios:

- **Protein structure prediction:** Accelerate inference for AlphaFold-style
  models whose trunk is built from triangle attention and triangle
  multiplicative update blocks.

- **Model training and fine-tuning:** Use the backward passes to train or
  fine-tune pair-representation models on AMD GPUs.

- **Porting cuequivariance-based code to AMD GPUs:** Run existing
  ``cuequivariance_torch`` code on ROCm through the
  ``hipequivariance-ops-torch`` compatibility package.

- **Cross-vendor deployments:** Use one kernel implementation across AMD and
  NVIDIA GPUs.

.. _geometric-nn-ops-support-matrix:

Operation support matrix
====================================================================

.. list-table::
   :header-rows: 1
   :align: left

   * - Operation
     - Forward
     - Backward
     - Notes
   * - ``triangle_attention``
     - ✅ Triton
     - ✅ Triton
     - Supports optional pair bias and boolean key mask.
   * - ``triangle_multiplicative_update``
     - ✅ Triton
     - ✅ PyTorch fallback
     - Native Triton backward pass is pending.
   * - ``attention_pair_bias``
     - ✅ PyTorch
     - ✅ PyTorch
     - No Triton kernel yet.

The Triton kernels accept ``fp32``, ``bf16``, and ``fp16`` inputs.

Usage example
====================================================================

Call the operations from the ``geometric_nn_ops_torch.triangle`` module. See
each function's docstring for the expected tensor layouts and other options.

.. code-block:: python

   from geometric_nn_ops_torch.triangle import (
       triangle_attention,
       triangle_multiplicative_update,
   )

   # q, k, v: [..., N_seq, N_res, H, D]
   o = triangle_attention(q, k, v, bias=pair_bias, mask=mask, scale=scale)

   # x: [*, N, N, C_z] pair representation
   z = triangle_multiplicative_update(x, direction="outgoing", mask=mask, **weights)
