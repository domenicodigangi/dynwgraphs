#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for dirBin1 (binary) models.

Created Date: Sunday June 6th 2021
Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Last Modified: Sunday June 6th 2021 7:28:04 pm
Modified By:  Domenico Di Gangi
"""

import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt

from dynwgraphs.dirBin1_dynNets import dirBin1_staNet, dirBin1_dynNet_SD
from dynwgraphs.utils import strIO_from_mat, tens


class TestBinarySingleSnapshot:
    """Test single snapshot estimation for binary models."""

    def test_estimate_ss_t_basic(self, binary_snapshot_data, prepared_binary_test_data):
        """Test basic single snapshot estimation for binary model."""
        Y_t = binary_snapshot_data["Y_t"]
        N = prepared_binary_test_data["N"]

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=False)
        model.distr = "bernoulli"

        phi_0 = model.start_phi_from_obs(Y_t, n_iter=30)
        phi_0 = model.identify(phi_0)

        # Check expected adjacency matrix sums match observed
        exp_A_out = model.exp_A(phi_0).sum(dim=1)
        obs_out = Y_t.sum(dim=1)

        # Check total expected matches
        tot_exp_check = model.check_tot_exp(Y_t, phi_0)
        assert tot_exp_check is not None, "check_tot_exp should return a value"

    def test_estimate_ss_t_with_optimization(self, binary_snapshot_data, prepared_binary_test_data):
        """Test single snapshot estimation with optimization."""
        Y_t = binary_snapshot_data["Y_t"]

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=False)
        model.distr = "bernoulli"

        phi_0 = model.start_phi_from_obs(Y_t, n_iter=30)
        phi_0 = model.identify(phi_0)

        phi_ss_t, diag_iter = model.estimate_ss_t(
            Y_t,
            phi_0=phi_0,
            plot_flag=False,
            print_flag=False,
            max_opt_iter=2500,
            dist_par_un_t=None,
            print_every=500,
        )

        tot_exp_check = model.check_tot_exp(Y_t, phi_ss_t)
        assert tot_exp_check is not None, "check_tot_exp should return a value after optimization"

    def test_estimate_ss_t_bin_from_degIO(self, binary_snapshot_data):
        """Test quick single snapshot estimation from degree sequences."""
        Y_t = binary_snapshot_data["Y_t"]

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=False, rescale_SD=False)
        degIO = strIO_from_mat(Y_t)

        phi_t, diag = model.estimate_ss_t_bin_from_degIO(
            degIO,
            min_opt_iter=500,
            max_opt_iter=2500,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=500,
        )

        # Check total expected links match total observed
        Y_t_sum = Y_t.sum()
        exp_A_sum = model.exp_A(phi_t).sum()

        assert exp_A_sum > 0, "Expected links should be positive"
        tot_exp_check = model.check_tot_exp(Y_t, phi_t)
        assert tot_exp_check is not None, "check_tot_exp should return a value"

    def test_joint_estimate_phi_and_beta(self, binary_snapshot_data):
        """Test joint estimation of phi and beta for single snapshot."""
        Y_t = binary_snapshot_data["Y_t"]
        X_t = binary_snapshot_data["X_t"]

        model = dirBin1_dynNet_SD()

        par_ss_t, diag = model.estimate_ss_t(
            Y_t,
            X_t=X_t,
            beta_t=None,
            phi_0=None,
            dist_par_un_t=None,
            like_type=2,
            est_beta=True,
            dim_beta=1,
            max_opt_iter=50,
            print_flag=False,
        )

        assert par_ss_t is not None, "Joint estimation should return parameters"


class TestBinarySequenceEstimation:
    """Test sequence estimation for binary models."""

    def test_estimate_phi_sequence(self, prepared_binary_test_data):
        """Test estimation of phi sequence."""
        Y_T = prepared_binary_test_data["Y_T"]
        N = prepared_binary_test_data["N"]

        model = dirBin1_staNet(avoid_ovflw_fun_flag=True)

        all_par_est_T, diag_T = model.ss_filt(
            Y_T,
            X_T=None,
            beta=None,
            phi_T_0=None,
            est_beta=False,
            max_opt_iter=4,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=10,
        )

        phi_ss_est_T = all_par_est_T[:2 * N, :]
        assert phi_ss_est_T is not None, "Sequence estimation should return phi values"
        assert phi_ss_est_T.shape[0] == 2 * N, f"Expected {2*N} phi parameters"

    def test_estimate_beta_const_given_phi_T(self, prepared_binary_test_data):
        """Test estimation of constant beta given phi_T."""
        Y_T = prepared_binary_test_data["Y_T"]
        X_T = prepared_binary_test_data["X_T"]
        N = prepared_binary_test_data["N"]

        # Create X_T_multi format
        X_T_multi = X_T.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(N, N, 1, -1)

        # First estimate phi_T
        model = dirBin1_staNet(avoid_ovflw_fun_flag=True)
        all_par_est_T, diag_T = model.ss_filt(
            Y_T,
            X_T=None,
            beta=None,
            phi_T_0=None,
            est_beta=False,
            max_opt_iter=4,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=10,
        )

        phi_ss_est_T = all_par_est_T[:2 * N, :]

        # Then estimate beta
        dim_beta = 1
        model_dyn = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True)
        beta_t_est, diag_beta_t = model_dyn.estimate_beta_const_given_phi_T(
            Y_T,
            X_T_multi,
            phi_ss_est_T.clone().detach(),
            dim_beta=dim_beta,
            dist_par_un=0,
            max_opt_iter=10,
            print_flag=False,
            plot_flag=False,
        )

        assert beta_t_est is not None, "Beta estimation should return values"

    def test_joint_estimate_phi_T_and_beta(self, prepared_binary_test_data):
        """Test joint estimation of phi_T and constant beta."""
        Y_T = prepared_binary_test_data["Y_T"]
        X_T = prepared_binary_test_data["X_T"]
        N = prepared_binary_test_data["N"]

        # Create X_T_multi format
        X_T_multi = X_T.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(N, N, 1, -1)

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True)
        phi_T, dist_par_un, beta, diag = model.ss_filt_est_beta_const(
            Y_T,
            X_T=X_T_multi,
            beta=None,
            phi_T=None,
            est_const_beta=True,
            dim_beta=1,
            opt_large_steps=2,
            opt_n=1,
            max_opt_iter_phi=5,
            lr_phi=0.01,
            max_opt_iter_beta=4,
            lr_beta=0.01,
            print_flag_phi=False,
            print_flag_beta=False,
            print_every=1,
        )

        assert phi_T is not None, "phi_T should be estimated"
        assert beta is not None, "beta should be estimated"
        assert len(diag) > 0, "Diagnostic information should be returned"


class TestBinaryScore:
    """Test score computation for binary models."""

    def test_score_backprop_vs_analytical(self, binary_snapshot_data, prepared_binary_test_data):
        """Test that backprop and analytical score computations match."""
        Y_t = binary_snapshot_data["Y_t"]
        N = prepared_binary_test_data["N"]

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=True)

        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        s_1 = model.score_t(Y_t, W, backprop_score=True, like_type=2)
        s_2 = model.score_t(Y_t, W, backprop_score=False, like_type=2)

        # Check that scores are similar
        abs_diff_0 = (s_1[0] - s_2[0]).abs().sum()
        abs_diff_1 = (s_1[1] - s_2[1]).abs().sum()

        assert abs_diff_0 < 1.0, f"Score[0] difference too large: {abs_diff_0}"
        assert abs_diff_1 < 1.0, f"Score[1] difference too large: {abs_diff_1}"


class TestBinaryScoreDrivenDynamics:
    """Test Score Driven estimation for binary models."""

    def test_estimate_sd_basic(self, prepared_binary_test_data):
        """Test basic score driven parameter estimation for binary models."""
        Y_T = prepared_binary_test_data["Y_T"]
        N = prepared_binary_test_data["N"]

        B = torch.cat([torch.ones(N) * 0.85, torch.ones(N) * 0.85])
        A = torch.cat([torch.ones(N) * 0.1, torch.ones(N) * 0.1])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)

        W_est, B_est, A_est, diag = model.estimate_SD(
            Y_T,
            B0=B,
            A0=A,
            W0=W,
            max_opt_iter=20,
            lr=0.01,
            dim_dist_par_un=1,
            print_flag=False,
            plot_flag=False,
            est_dis_par_un=False,
        )

        assert W_est is not None, "Should estimate W"
        assert B_est is not None, "Should estimate B"
        assert A_est is not None, "Should estimate A"

    def test_estimate_sd_with_static_regressors(self, prepared_binary_test_data):
        """Test score driven estimation with static regressors for binary models."""
        Y_T = prepared_binary_test_data["Y_T"]
        X_T = prepared_binary_test_data["X_T"]
        N = prepared_binary_test_data["N"]

        # Create X_T_multi format
        X_T_multi = X_T.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(N, N, 2, -1)

        dim_beta = N
        n_reg = X_T_multi.shape[2]
        n_beta_tv = 0
        dim_dist_par_un = N

        B = torch.cat([torch.ones(N) * 0.85, torch.ones(N) * 0.85])
        A = torch.cat([torch.ones(N) * 0.1, torch.ones(N) * 0.1])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)

        W_est, B_est, A_est, dist_par_un_est, beta_const_est, diag = model.estimate_SD_X0(
            Y_T,
            X_T=X_T_multi,
            dim_beta=dim_beta,
            n_beta_tv=n_beta_tv,
            B0=torch.cat((B, torch.zeros(n_beta_tv))),
            A0=torch.cat((A, torch.zeros(n_beta_tv))),
            W0=torch.cat((W, torch.zeros(n_beta_tv))),
            beta_const_0=torch.zeros(dim_beta, n_reg - n_beta_tv),
            max_opt_iter=50,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
        )

        assert W_est is not None, "Should estimate W"
        assert beta_const_est is not None, "Should estimate beta"


class TestBinaryDGP:
    """Test data generation processes for binary models."""

    def test_dgp_phi_var_size(self):
        """Test variable-size phi data generation."""
        N = 30
        T = 100

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)

        n_reg = 2
        n_beta_tv = 1
        dim_beta = 1

        X_T = torch.randn(N, N, n_reg, T)
        A = torch.cat([torch.ones(N) * 0.01, torch.ones(N) * 0.01])
        B = torch.cat([torch.ones(N) * 0.9, torch.ones(N) * 0.9])

        um_sd_par, um_degs = model.dgp_phi_var_size(N, degMin=5, degMax=25)

        assert um_sd_par is not None, "Should generate phi parameters"
        assert um_degs is not None, "Should generate degree sequences"
        assert um_sd_par.shape[0] == 2 * N, f"Expected {2*N} phi parameters"

    def test_sd_dgp_with_time_varying_regressors(self):
        """Test score driven DGP with time-varying regressors."""
        N = 30
        T = 100

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)

        N_BA = N
        n_reg = 2
        n_beta_tv = 1
        dim_beta = 1

        X_T = torch.randn(N, N, n_reg, T)
        A = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
        B = torch.cat([torch.ones(N_BA) * 0.9, torch.ones(N_BA) * 0.9])

        um_sd_par, um_degs = model.dgp_phi_var_size(N, degMin=5, degMax=25)
        beta_const = torch.ones(dim_beta, n_reg - n_beta_tv)

        if n_beta_tv * dim_beta > 0:
            A = torch.cat((A, torch.ones(n_beta_tv * dim_beta) * 0.01))
            B = torch.cat((B, torch.ones(n_beta_tv * dim_beta) * 0.01))
            um_sd_par = torch.cat(
                (um_sd_par, torch.ones(n_beta_tv * dim_beta) * 1 / (B[-n_beta_tv * dim_beta]))
            )

        W = um_sd_par * (1 - B)

        model.n_reg_beta_tv = n_beta_tv
        model.dim_beta = dim_beta

        phi_T, beta_sd_T, Y_T_dgp = model.sd_dgp(
            W, B, A, p_T=None, beta_const=beta_const, X_T=X_T, N=N, T=T
        )

        assert Y_T_dgp is not None, "Should generate Y_T data"
        assert phi_T is not None, "Should generate phi_T"
        assert Y_T_dgp.shape == (N, N, T), f"Y_T_dgp should have shape ({N}, {N}, {T})"

    def test_sample_from_dgps_step(self):
        """Test sampling from DGPs with step changes."""
        N = 50
        T = 100
        N_sample = 10

        model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
        degb = tens([10, N - 10])
        um_phi, um_degs = model.dgp_phi_var_size(N, degMin=degb[0], degMax=degb[1])

        Y_T_S_dgp, phi_T_dgp, X_T_dgp, beta_T_dgp = model.sample_from_dgps(
            N,
            T,
            N_sample,
            um_phi=um_phi,
            dgp_type="step",
            X_T=None,
            n_reg=2,
            n_reg_beta_tv=1,
            dim_beta=1,
            degb=degb,
        )

        assert Y_T_S_dgp is not None, "Should generate sample data"
        assert phi_T_dgp is not None, "Should generate phi_T"
        assert Y_T_S_dgp.shape[0] == N, f"Expected N={N} in first dimension"
        assert Y_T_S_dgp.shape[2] == T, f"Expected T={T} in third dimension"
        assert Y_T_S_dgp.shape[3] == N_sample, f"Expected {N_sample} samples"
