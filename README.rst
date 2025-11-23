==========
dynwgraphs
==========


.. image:: https://img.shields.io/pypi/v/dynwgraphs.svg
        :target: https://pypi.python.org/pypi/dynwgraphs

.. image:: https://img.shields.io/travis/digangidomenico/dynwgraphs.svg
        :target: https://travis-ci.com/digangidomenico/dynwgraphs

.. image:: https://readthedocs.org/projects/dynwgraphs/badge/?version=latest
        :target: https://dynwgraphs.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


A PyTorch-based package for modeling dynamical weighted and binary graphs (complex networks) using zero-augmented statistical models with score-driven dynamics.

* Free software: MIT license
* Python 3.8+

Features
--------

* **Static Network Models**: Single-snapshot parameter estimation for directed graphs
* **Dynamic Network Models**: Score-driven dynamics for time-varying parameters
* **Binary Networks**: Bernoulli-based models for presence/absence of edges
* **Weighted Networks**: Gamma and Log-Normal distributions for edge weights
* **Zero-Augmented Models**: Handle sparse networks with many zero-valued edges
* **External Regressors**: Include time-varying and static covariates
* **PyTorch Integration**: GPU acceleration and automatic differentiation
* **Flexible Optimization**: Multiple optimizers (Adam, SGD, L-BFGS, hypergradient-based)
* **Out-of-Sample Evaluation**: Filtering and forecasting capabilities

Installation
------------

Basic installation:

.. code-block:: bash

    pip install dynwgraphs

For development:

.. code-block:: bash

    git clone https://github.com/digangidomenico/dynwgraphs.git
    cd dynwgraphs
    pip install -e .
    pip install -r requirements_dev.txt

Quick Start
-----------

**Static Weighted Network Model**

.. code-block:: python

    import torch
    from dynwgraphs import dirSpW1_staNet
    from dynwgraphs.utils import tens

    # Load or create network data
    # Y_T: N×N×T tensor of weighted adjacency matrices
    Y_T = torch.rand(10, 10, 50)  # Example: 10 nodes, 50 time periods

    # Initialize model
    model = dirSpW1_staNet()
    model.distr = 'gamma'  # or 'lognormal'

    # Estimate parameters
    model.init_all_par_sequences(Y_T, T_train=Y_T.shape[2])
    phi_T, diagnostics = model.ss_filt(
        Y_T,
        phi_T_0=model.start_phi_from_obs_T(Y_T),
        max_opt_iter=100
    )

**Dynamic Network with Score-Driven Dynamics**

.. code-block:: python

    from dynwgraphs import dirSpW1_dynNet_SD

    # Initialize score-driven model
    model_sd = dirSpW1_dynNet_SD()
    model_sd.distr = 'gamma'

    # Set up model
    model_sd.init_all_par_sequences(Y_T, T_train=40)

    # Estimate with score-driven filtering
    sd_par_est, diagnostics = model_sd.estimate_SD(
        Y_T,
        max_opt_iter=100,
        print_flag=True
    )

    # Forecast out-of-sample
    forecast = model_sd.get_SD_forecast(n_steps_ahead=5)

**Binary Network Model**

.. code-block:: python

    from dynwgraphs import dirBin1_dynNet_SD

    # Binary adjacency matrix (0/1)
    A_T = (Y_T > 0).float()

    # Initialize binary model
    model_bin = dirBin1_dynNet_SD()
    model_bin.init_all_par_sequences(A_T, T_train=40)

    # Estimate
    par_est, diag = model_bin.estimate_SD(A_T, max_opt_iter=100)

Model Types
-----------

**Weighted Models (dirSpW1)**

* ``dirSpW1_staNet``: Static network (constant parameters)
* ``dirSpW1_dynNet_SD``: Dynamic network with score-driven parameters

**Binary Models (dirBin1)**

* ``dirBin1_staNet``: Static binary network
* ``dirBin1_dynNet_SD``: Dynamic binary network

**Distributions**

* ``gamma``: For positive continuous edge weights
* ``lognormal``: For positive continuous edge weights with different tail behavior

Dependencies
------------

Core:
    * Python >= 3.8
    * PyTorch >= 1.9.0
    * NumPy >= 1.19
    * Pandas >= 1.2
    * SciPy >= 1.6
    * scikit-learn >= 0.24
    * statsmodels >= 0.12
    * matplotlib >= 3.3

Optional:
    * Adahessian optimizer (git submodule)
    * Tobit models (git submodule)

Testing
-------

Run tests with pytest:

.. code-block:: bash

    pytest tests/

With coverage:

.. code-block:: bash

    pytest tests/ --cov=dynwgraphs --cov-report=html

Documentation
-------------

Full documentation coming soon at https://dynwgraphs.readthedocs.io

Contributing
------------

Contributions are welcome! Please see CONTRIBUTING.rst for guidelines.

Development setup:

.. code-block:: bash

    pip install -r requirements_dev.txt
    pre-commit install  # Install git hooks

Run code quality checks:

.. code-block:: bash

    black dynwgraphs tests        # Format code
    isort dynwgraphs tests        # Sort imports
    flake8 dynwgraphs tests       # Lint
    pytest tests/ --cov           # Test with coverage

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

    @software{dynwgraphs2021,
      author = {Di Gangi, Domenico},
      title = {DynWGraphs: Dynamic Weighted Graphs Modeling},
      year = {2021},
      url = {https://github.com/digangidomenico/dynwgraphs}
    }

License
-------

MIT License - see LICENSE file for details

Authors
-------

* **Domenico Di Gangi** - *Initial work* - digangidomenico@gmail.com

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
