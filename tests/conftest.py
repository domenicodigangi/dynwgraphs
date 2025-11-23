#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared pytest fixtures for dynwgraphs tests.

@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
"""

import os
import numpy as np
import torch
import pytest
import matplotlib
import matplotlib.pyplot as plt

from dynwgraphs.utils import tens
from dynwgraphs.utils.dgps import get_test_w_mat

# Set matplotlib to non-interactive backend for tests
matplotlib.use('Agg')


@pytest.fixture(scope="session")
def test_data_path():
    """Return the path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session")
def test_data_file(test_data_path):
    """Return the path to the test data NPZ file."""
    return os.path.join(test_data_path, "dir_w_test_data.npz")


@pytest.fixture(scope="session")
def test_data_exists(test_data_file):
    """Check if test data file exists."""
    return os.path.exists(test_data_file)


@pytest.fixture
def synthetic_test_data():
    """Generate synthetic test data using dgps module."""
    N = 10
    T = 30

    # Generate reference matrix
    Y_ref = get_test_w_mat(avg_weight=1e5, T_avg_data=10, N=N)

    # Create time series by adding random variations
    Y_T = torch.zeros(N, N, T)
    for t in range(T):
        noise = torch.randn(N, N) * 0.1 * Y_ref
        Y_T[:, :, t] = Y_ref + noise
        Y_T[:, :, t] = torch.clamp(Y_T[:, :, t], min=0)

    # Create scalar regressors
    X_T = torch.randn(T) * 100 + 1000

    return {
        "Y_T": Y_T.numpy(),
        "X_T": X_T.numpy(),
        "N": N,
        "T": T
    }


@pytest.fixture
def loaded_test_data(test_data_exists, test_data_file, synthetic_test_data):
    """
    Load test data from file if it exists, otherwise use synthetic data.

    Returns a dictionary with Y_T, X_T, N, and T.

    SECURITY: Uses allow_pickle=False to prevent arbitrary code execution
    when loading numpy files.
    """
    if test_data_exists:
        # SECURITY: Disable pickle to prevent code execution
        data = np.load(test_data_file, allow_pickle=False)
        N = data["Y_T"].shape[0]
        T = data["Y_T"].shape[2]
        return {
            "Y_T": data["Y_T"],
            "X_T": data["X_T"],
            "N": N,
            "T": T
        }
    else:
        return synthetic_test_data


@pytest.fixture
def prepared_test_data(loaded_test_data):
    """
    Prepare test data in the format expected by dirSpW1 models.

    Returns tensors: Y_T, X_scalar_T, X_matrix_T, X_T_multi, N, T
    """
    unit_measure = 1e5
    Y_T = tens(loaded_test_data["Y_T"]) / unit_measure

    N = Y_T.shape[0]
    T = Y_T.shape[2]

    X_scalar_T = tens(loaded_test_data["X_T"]).unsqueeze(1).unsqueeze(1).permute((3, 2, 1, 0))
    X_matrix_T = Y_T[:, :, :-1].log().unsqueeze(2)
    X_matrix_T[~torch.isfinite(X_matrix_T)] = 0

    X_T_multi = X_matrix_T.repeat_interleave(2, dim=2)
    X_T_multi[:, :, 1, :] += 1

    return {
        "Y_T": Y_T,
        "X_scalar_T": X_scalar_T,
        "X_matrix_T": X_matrix_T,
        "X_T_multi": X_T_multi,
        "N": N,
        "T": T
    }


@pytest.fixture
def prepared_binary_test_data(loaded_test_data):
    """
    Prepare test data in binary format for dirBin1 models.

    Returns tensors: Y_T (binary), X_T, N, T
    """
    Y_T = tens(loaded_test_data["Y_T"] > 0)
    X_T = tens(loaded_test_data["X_T"])

    N = Y_T.shape[0]
    T = Y_T.shape[2]

    return {
        "Y_T": Y_T,
        "X_T": X_T,
        "N": N,
        "T": T
    }


@pytest.fixture
def snapshot_data(prepared_test_data):
    """
    Extract a single time snapshot for single-snapshot tests.

    Returns: Y_t, X_t, A_t, Y_tp1, t
    """
    Y_T = prepared_test_data["Y_T"]
    X_matrix_T = prepared_test_data["X_matrix_T"]
    T = prepared_test_data["T"]

    t = T - 12
    Y_t = Y_T[:, :, t]
    X_t = X_matrix_T[:, :, :, t]
    A_t = Y_t > 0
    Y_tp1 = Y_T[:, :, t + 1]

    return {
        "Y_t": Y_t,
        "X_t": X_t,
        "A_t": A_t,
        "Y_tp1": Y_tp1,
        "t": t
    }


@pytest.fixture
def binary_snapshot_data(prepared_binary_test_data):
    """
    Extract a single time snapshot for binary model tests.

    Returns: Y_t, X_t, t
    """
    Y_T = prepared_binary_test_data["Y_T"]
    X_T = prepared_binary_test_data["X_T"]

    t = 0
    Y_t = Y_T[:, :, t]
    X_t = X_T[t].unsqueeze(1).unsqueeze(1)

    return {
        "Y_t": Y_t,
        "X_t": X_t,
        "t": t
    }


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all matplotlib plots after each test."""
    yield
    plt.close('all')
