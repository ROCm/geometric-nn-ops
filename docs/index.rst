.. meta::
  :description: geometric-nn-ops documentation
  :keywords: geometric-nn-ops, ROCm, documentation, Triton, protein folding, triangle attention, GPU

.. _geometric-nn-ops-documentation-index:

********************************************************************
geometric-nn-ops on ROCm documentation
********************************************************************

Accelerate geometry-aware neural networks, such as protein structure prediction
models, with portable Triton GPU kernels that run on AMD GPUs with ROCm.

geometric-nn-ops is a library of vendor-neutral GPU kernels, with both forward
and backward passes, for geometry-aware neural networks. The kernels are written
in `Triton <https://triton-lang.org/>`__, so the same source runs on AMD GPUs
with ROCm and on NVIDIA GPUs with CUDA.

The first kernel family, ``geometric_nn_ops_torch.triangle``, covers the triangle
operations at the core of AlphaFold-style protein folding models:
``triangle_attention``, ``triangle_multiplicative_update``, and
``attention_pair_bias``. The kernels are exposed through a PyTorch API, and an
optional compatibility package, ``hipequivariance-ops-torch``, lets code written
against NVIDIA's ``cuequivariance_ops_torch`` run on AMD GPUs without source
changes.

.. note::

  geometric-nn-ops is under active development, and some operations do not yet
  have a native Triton kernel for every pass. For the current support status, see :ref:`geometric-nn-ops-support-matrix`.

The geometric-nn-ops project is located in `ROCm/geometric-nn-ops <https://github.com/ROCm/geometric-nn-ops>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Introduction

    * :doc:`What is geometric-nn-ops? <about/what-is-geometric-nn-ops>`

  .. grid-item-card:: Install

    * :doc:`Install geometric-nn-ops <install/geometric-nn-ops-install>`

To contribute to geometric-nn-ops, refer to `CONTRIBUTING.md <https://github.com/ROCm/geometric-nn-ops/blob/main/CONTRIBUTING.md>`_.

You can find licensing information on the :doc:`Licensing <license>` page.
