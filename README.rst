==========
dynwgraphs
==========


.. image:: https://img.shields.io/pypi/v/dynwgraphs.svg
        :target: https://pypi.python.org/pypi/dynwgraphs

.. image:: https://readthedocs.org/projects/dynwgraphs/badge/?version=latest
        :target: https://dynwgraphs.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



A package to model dynamical weighted, possibly sparse, graphs a.k.a. known as complex networks


* Free software: MIT license


.. warning::

   **Maintenance Notice (March 2025):** This repository has been updated using Claude Code
   after a long period since original publication. The changes include bug fixes, security
   improvements, dependency updates, and CI modernization. **These changes have not been
   manually tested.** Please review and test thoroughly before using in production.


Features
--------

* Zero-augmented distribution models for sparse weighted directed networks
* Support for both binary (Bernoulli) and weighted (Gamma, Log-normal) link distributions
* Static single-snapshot and dynamic score-driven parameter estimation
* External regressors support with time-varying or constant coefficients
* Multiple optimization backends (Adam, SGD, L-BFGS, hypergradient methods)
* PyTorch-based with automatic differentiation for likelihood optimization

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
