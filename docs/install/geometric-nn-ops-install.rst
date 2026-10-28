.. meta::
  :description: installing geometric-nn-ops for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, geometric-nn-ops, Triton, PyTorch, wheel

.. _geometric-nn-ops-on-rocm-installation:

********************************************************************
geometric-nn-ops on ROCm installation
********************************************************************

This topic discusses how to install geometric-nn-ops using the following options:

* :ref:`geometric-nn-ops-wheel-install` **(recommended)**
* :ref:`geometric-nn-ops-editable-install`

.. note::

  The geometric-nn-ops packages are not yet published on PyPI. Until they are,
  build and install them from the source repository using one of the options on
  this page.

System requirements
====================================================================

To use geometric-nn-ops 1.0.0, you need the following prerequisites:

.. list-table::
   :header-rows: 1

   * - ROCm version
     - Ubuntu version
     - Python version
     - AMD Instinct™ GPU
     - PyTorch version
     - Triton version
   * - 10.0.0
     - 24.04
     - 3.12
     - MI300X, MI325X, and MI355X
     - 2.13.0
     - 3.7.1
   * - 10.1.0
     - 24.04
     - 3.12
     - MI300X, MI325X, and MI355X
     - 2.13.0
     - 3.7.1

The installation steps on this page are validated with the
``rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0``
and
``rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0``
Docker images, which provide ROCm, PyTorch, and Triton.

.. _geometric-nn-ops-install-options:

Install geometric-nn-ops
====================================================================

geometric-nn-ops consists of two pure-Python packages that contain Triton
kernels:

- ``geometric-nn-ops-torch``: the kernels and their PyTorch API (required).
- ``hipequivariance-ops-torch``: an optional ``cuequivariance_ops_torch``
  compatibility package. See :ref:`geometric-nn-ops-shim-install`.

Because the packages are pure Python, building them does not require the ROCm
compiler toolchain. PyTorch and Triton are provided with your existing ROCm
environment.

To install geometric-nn-ops on ROCm, you have the following options:

* :ref:`geometric-nn-ops-wheel-install` **(recommended)**
* :ref:`geometric-nn-ops-editable-install`

.. _geometric-nn-ops-wheel-install:

Build and install the wheel in a ROCm PyTorch Docker container
-------------------------------------------------------------------

Docker is the recommended method to set up a geometric-nn-ops environment, as it
avoids dependency conflicts. The ROCm PyTorch image already includes ROCm,
PyTorch, and Triton.

1. Pull the ROCm-enabled PyTorch Docker image for your ROCm version.

   .. tab-set::

      .. tab-item:: ROCm 10.0.0
         :sync: rocm-10-0-0

         .. code-block:: bash

            docker pull rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

      .. tab-item:: ROCm 10.1.0
         :sync: rocm-10-1-0

         .. code-block:: bash

            docker pull rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

2. Start a Docker container using the same image.

   .. tab-set::

      .. tab-item:: ROCm 10.0.0
         :sync: rocm-10-0-0

         .. code-block:: bash

            docker run -it --rm \
            --network=host --device=/dev/kfd \
            --device=/dev/dri --group-add video \
            --name=my_geometric_nn_ops \
            --security-opt seccomp=unconfined \
            --ipc=host \
            rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

      .. tab-item:: ROCm 10.1.0
         :sync: rocm-10-1-0

         .. code-block:: bash

            docker run -it --rm \
            --network=host --device=/dev/kfd \
            --device=/dev/dri --group-add video \
            --name=my_geometric_nn_ops \
            --security-opt seccomp=unconfined \
            --ipc=host \
            rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

3. Inside the container, clone the geometric-nn-ops repository.

   .. code-block:: bash

      git clone https://github.com/ROCm/geometric-nn-ops geometric-nn-ops
      cd geometric-nn-ops
      git checkout release/v1.0.0

4. Build the ``geometric-nn-ops-torch`` wheel.

   .. code-block:: bash

      pip wheel . --no-deps -w ./dist/

   ``pip`` downloads the ``hatchling`` build back end during the build, so the
   container needs network access.

5. Install the wheel.

   .. code-block:: bash

      pip install --no-deps ./dist/geometric_nn_ops_torch-*.whl

.. important::

  Always pass ``--no-deps`` when you build or install the geometric-nn-ops packages.
  ``pip`` checks the ``torch`` and ``triton`` requirements by package name, and ROCm
  builds of Triton are also shipped as ``triton-rocm`` or ``pytorch-triton-rocm``.
  When a requirement looks unmet, by package name or by version, ``pip`` installs
  PyPI's builds over the ROCm ones: a ``torch`` built for CUDA, or a ``triton``
  that need not match your PyTorch.

.. _geometric-nn-ops-editable-install:

Install from source for development
------------------------------------------------------------

If you plan to modify geometric-nn-ops, install it in editable mode so that
changes to the source tree take effect without reinstalling. Use an environment
that already has ROCm, a ROCm build of PyTorch, and Triton, such as the
``rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0``
or
``rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0``
container from the previous section.

1. Clone the geometric-nn-ops repository.

   .. code-block:: bash

      git clone https://github.com/ROCm/geometric-nn-ops geometric-nn-ops
      cd geometric-nn-ops
      git checkout release/v1.0.0

2. Install the ``geometric-nn-ops-torch`` package in editable mode.

   .. code-block:: bash

      pip install --no-deps -e .

.. _geometric-nn-ops-shim-install:

Install the cuequivariance compatibility package (optional)
====================================================================

The ``hipequivariance-ops-torch`` package implements the ``cuequivariance_ops_torch`` module API and routes calls to the geometric-nn-ops
kernels. With it installed, unmodified code that uses ``cuequivariance_torch``,
for example ``from cuequivariance_torch.primitives.triangle import
triangle_attention``, runs on AMD GPUs.

This package is optional and is never installed implicitly. Install it only if
your code already depends on cuequivariance. Otherwise, use
``geometric_nn_ops_torch`` directly.

.. warning::

  Do not install the ``cuequivariance-ops-torch``, ``cuequivariance-ops-torch-cu12``,
  or ``cuequivariance-ops-torch-cu13`` packages in the same environment. They
  provide the same ``cuequivariance_ops_torch`` module, and the shim raises an
  error at import time if it detects any of them.

1. Install ``geometric-nn-ops-torch`` using one of the options in
   :ref:`geometric-nn-ops-install-options`.

2. From the root of the cloned repository, install the
   ``hipequivariance-ops-torch`` package.

   .. tab-set::

      .. tab-item:: Wheel

         .. code-block:: bash

            pip wheel ./hipequivariance_ops_torch --no-deps -w ./dist/
            pip install --no-deps ./dist/hipequivariance_ops_torch-*.whl

      .. tab-item:: Editable

         .. code-block:: bash

            pip install --no-deps -e ./hipequivariance_ops_torch

3. Install the cuequivariance Python front end. The shim is designed for
   version 0.9.1 and warns at import time if a different version of
   ``cuequivariance-torch`` is installed.

   .. code-block:: bash

      pip install "cuequivariance==0.9.1" "cuequivariance-torch==0.9.1"

Test the geometric-nn-ops installation
====================================================================

1. Verify that PyTorch can see your AMD GPU.

   .. code-block:: bash

      python3 -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} hip={torch.version.hip}")'

   The output should show ``cuda_available=True``, a ``device_count`` of at
   least ``1``, and the HIP version of your PyTorch build.

2. Verify that geometric-nn-ops is installed correctly. If you installed the
   wheel, ``-P`` keeps ``python3`` from importing the source tree in the current
   directory instead of the installed package.

   .. code-block:: bash

      python3 -P -c "import geometric_nn_ops_torch; print(geometric_nn_ops_torch.__version__)"

   Expected output:

   .. code-block:: text

      1.0.0

3. If you installed the ``hipequivariance-ops-torch`` compatibility package,
   verify that the ``cuequivariance_ops_torch`` module resolves to the shim.

   .. code-block:: bash

      python3 -c "import cuequivariance_ops_torch; print(cuequivariance_ops_torch.__version__)"

   Expected output:

   .. code-block:: text

      1.0.0

If you see the version strings above, geometric-nn-ops ``1.0.0`` has been
installed successfully.

Run the tests
---------------------------------------

The test suite runs on the GPU and checks the kernels against reference
implementations.

Install ``pytest`` on its own rather than through the packages' ``[test]``
extras. This ensures that ``pip`` doesn't pull in other dependencies over your
``--no-deps`` installation.

Run the tests from the root of the cloned repository.

.. code-block:: bash

   pip install "pytest==9.0.3"
   python3 -m pytest -vrxs tests

The tests import ``geometric_nn_ops_torch`` from the source tree in the current
directory, not from an installed wheel.

.. note::

  Two ``fp16`` tests are expected to fail and appear as ``2 xfailed`` in the
  ``pytest`` summary. They cover fully masked rows, where softmax is undefined,
  and do not affect normal use.

Tests that compare against cuequivariance are skipped unless the
``cuequivariance-torch`` package is installed. Tests marked ``large`` are
deselected by default because they need datacenter-class GPU memory and take a
long time to run. To include them, add ``-m ""`` to the ``pytest`` command
above.

Use the following command to run tests for the ``hipequivariance-ops-torch`` package:

.. code-block:: bash

   python3 -m pytest -vrxs hipequivariance_ops_torch/tests
