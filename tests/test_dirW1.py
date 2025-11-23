#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for dirSpW1_dynNet_SD model.

@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Thursday June 10th 2021
Last Modified: Thursday June 10th 2021 12:01:25 pm, by Domenico Di Gangi
"""

import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt

from dynwgraphs import dirSpW1_dynNet_SD
from dynwgraphs.dirGraphs1_dynNets import dirSpW1_staNet
from dynwgraphs.utils import tens


class TestPhiStartingPoints:
    """Test starting point estimation for phi parameter."""

    def test_start_phi_from_obs(self, snapshot_data, prepared_test_data):
        """Test that starting phi values produce reasonable conditional expectations."""
        Y_t = snapshot_data["Y_t"]
        N = prepared_test_data["N"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        model.distr = 'gamma'
        phi_0 = model.start_phi_from_obs(Y_t)

        # Check that conditional expectation is reasonably close to observed values
        EY = model.cond_exp_Y(phi_0)
        rel_error_active = (EY[Y_t > 0].sum() - Y_t.sum()) / Y_t.sum()
        rel_error_total = (EY.sum() - Y_t.sum()) / Y_t.sum()

        # Starting values should have relative error less than 50%
        assert abs(rel_error_active.item()) < 0.5, f"Active link relative error too high: {rel_error_active}"
        assert abs(rel_error_total.item()) < 0.5, f"Total relative error too high: {rel_error_total}"


class TestSingleSnapshotEstimates:
    """Test single snapshot (ss_t) estimation methods."""

    def test_estimate_ss_t_converges(self, snapshot_data, prepared_test_data):
        """Test that single snapshot phi estimation converges."""
        Y_t = snapshot_data["Y_t"]
        N = prepared_test_data["N"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        model.distr = 'gamma'

        # Initialize phi
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        phi_0 = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        # Run estimation multiple times
        for i in range(3):
            phi_ss_t, diag_iter = model.estimate_ss_t(
                Y_t,
                phi_0=phi_0,
                plot_flag=False,
                print_flag=False,
                max_opt_iter=500,
                dist_par_un_t=torch.zeros(1)
            )
            phi_0 = phi_ss_t

            # Check total expected matches total observed
            tot_exp_check = model.check_tot_exp(Y_t, phi_ss_t)
            assert tot_exp_check is not None, f"check_tot_exp returned None at iteration {i}"

        # Check that fitted values are reasonable
        beta_t = torch.zeros(N)
        X_t = snapshot_data["X_t"]
        A_t = Y_t > 0
        EYcond_mat = model.cond_exp_Y(phi_ss_t, beta=beta_t, X_t=X_t)

        # Most active links should have positive expected values
        assert EYcond_mat[A_t].min() > 0, "Expected values for active links should be positive"

        # Check distribution of observed/expected ratios
        tmp = Y_t[A_t] / EYcond_mat[A_t]
        mean_ratio = tmp.mean()
        assert 0.5 < mean_ratio < 2.0, f"Mean observed/expected ratio out of range: {mean_ratio}"

    def test_joint_estimate_phi_and_dist_par(self, snapshot_data, prepared_test_data):
        """Test joint estimation of phi and distribution parameters."""
        Y_t = snapshot_data["Y_t"]
        X_t = snapshot_data["X_t"]
        t = snapshot_data["t"]
        N = prepared_test_data["N"]
        X_T_multi = prepared_test_data["X_T_multi"]
        Y_T = prepared_test_data["Y_T"]

        model = dirSpW1_dynNet_SD()
        dim_dist_par_un = N
        dim_beta = N
        n_reg = X_T_multi.shape[2]
        beta = torch.zeros(dim_beta, n_reg) * 1
        phi_T_0 = model.start_phi_from_obs_T(Y_T)
        phi_t_0 = phi_T_0[:, t]

        par_ss_t, diag = model.estimate_ss_t(
            Y_t,
            X_t=X_T_multi[:, :, :, t],
            beta_t=beta,
            phi_0=phi_t_0,
            dist_par_un_t=model.dist_par_un_start_val(dim_dist_par_un),
            like_type=2,
            est_dist_par=True,
            dim_dist_par_un=N,
            max_opt_iter=50,
            print_flag=False,
            print_every=1
        )

        # Check that parameters were estimated
        assert par_ss_t is not None, "Estimation should return parameters"

    def test_joint_estimate_phi_dist_par_and_beta(self, snapshot_data, prepared_test_data):
        """Test joint estimation of phi, distribution parameters, and beta."""
        Y_t = snapshot_data["Y_t"]
        X_t = snapshot_data["X_t"]
        N = prepared_test_data["N"]

        model = dirSpW1_dynNet_SD()

        par_ss_t, diag = model.estimate_ss_t(
            Y_t,
            X_t=X_t,
            beta_t=None,
            phi_0=None,
            dist_par_un_t=None,
            like_type=2,
            est_dist_par=True,
            dim_dist_par_un=N,
            est_beta=True,
            dim_beta=1,
            max_opt_iter=50,
            print_flag=False
        )

        assert par_ss_t is not None, "Joint estimation should return parameters"


class TestSequenceEstimation:
    """Test estimation of sequences over time."""

    def test_estimate_phi_sequence_static_dist_par(self, prepared_test_data):
        """Test estimation of phi sequence with static distribution parameters."""
        Y_T = prepared_test_data["Y_T"]
        N = prepared_test_data["N"]

        model = dirSpW1_staNet(avoid_ovflw_fun_flag=True)
        dim_dist_par_un = N
        phi_T_0 = model.start_phi_from_obs_T(Y_T)
        dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)

        all_par_est_T, diag_T = model.ss_filt(
            Y_T,
            X_T=None,
            beta=None,
            phi_T_0=phi_T_0,
            dist_par_un=dist_par_un,
            est_dist_par=False,
            est_beta=False,
            max_opt_iter=4,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=1
        )

        assert all_par_est_T is not None, "Sequence estimation should return parameters"
        assert all_par_est_T.shape[0] == 2 * N, f"Expected {2*N} parameters, got {all_par_est_T.shape[0]}"

    def test_estimate_phi_sequence_with_beta(self, prepared_test_data):
        """Test estimation of phi sequence with static distribution parameters and beta."""
        Y_T = prepared_test_data["Y_T"]
        X_T_multi = prepared_test_data["X_T_multi"]
        N = prepared_test_data["N"]
        T = prepared_test_data["T"]

        model = dirSpW1_staNet(avoid_ovflw_fun_flag=True)
        dim_dist_par_un = N
        dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)
        dim_beta = N
        n_reg = X_T_multi.shape[2]
        beta = torch.zeros(dim_beta, n_reg) * 1
        phi_T_0 = torch.randn(2 * N, T) * 0.0001

        all_par_est_T, diag_T = model.ss_filt(
            Y_T,
            X_T=X_T_multi,
            beta=beta,
            phi_T_0=phi_T_0,
            dist_par_un=dist_par_un,
            est_dist_par=False,
            est_beta=False,
            max_opt_iter=20,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=1
        )

        assert all_par_est_T is not None, "Sequence estimation with beta should return parameters"


class TestBetaEstimation:
    """Test beta coefficient estimation."""

    def test_estimate_beta_const_given_phi_T(self, prepared_test_data):
        """Test estimation of constant beta given phi_T and dist_par."""
        Y_T = prepared_test_data["Y_T"]
        X_T_multi = prepared_test_data["X_T_multi"]
        N = prepared_test_data["N"]

        # First estimate phi_T
        model = dirSpW1_staNet(avoid_ovflw_fun_flag=True)
        dim_dist_par_un = N
        dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)
        all_par_est_T, diag_T = model.ss_filt(
            Y_T,
            X_T=None,
            beta=None,
            phi_T_0=None,
            dist_par_un=dist_par_un,
            est_dist_par=False,
            est_beta=False,
            max_opt_iter=4,
            opt_n=1,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=10
        )

        phi_ss_est_T = all_par_est_T[:2 * N, :]

        # Then estimate beta
        dim_beta = 1
        model_dyn = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        beta_t_est, diag_beta_t = model_dyn.estimate_beta_const_given_phi_T(
            Y_T,
            X_T_multi,
            phi_ss_est_T,
            dim_beta=dim_beta,
            dist_par_un=dist_par_un,
            max_opt_iter=10,
            print_flag=False,
            plot_flag=False
        )

        assert beta_t_est is not None, "Beta estimation should return values"

    def test_joint_estimate_phi_T_beta_and_dist_par_const(self, prepared_test_data):
        """Test joint estimation of phi_T, constant beta, and distribution parameters."""
        Y_T = prepared_test_data["Y_T"]
        X_T_multi = prepared_test_data["X_T_multi"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        phi_T, dist_par_un, beta, diag = model.ss_filt_est_beta_dist_par_const(
            Y_T,
            X_T=X_T_multi,
            beta=None,
            phi_T=None,
            dist_par_un=None,
            like_type=2,
            est_const_dist_par=True,
            dim_dist_par_un=1,
            est_const_beta=True,
            dim_beta=1,
            opt_large_steps=4,
            opt_n=1,
            max_opt_iter_phi=15,
            lr_phi=0.01,
            max_opt_iter_dist_par=15,
            lr_dist_par=0.01,
            max_opt_iter_beta=4,
            lr_beta=0.01,
            print_flag_phi=False,
            print_flag_dist_par=False,
            print_flag_beta=False,
            print_every=1
        )

        assert phi_T is not None, "phi_T should be estimated"
        assert dist_par_un is not None, "dist_par_un should be estimated"
        assert beta is not None, "beta should be estimated"
        assert len(diag) > 0, "Diagnostic information should be returned"


class TestScore:
    """Test score function computation."""

    def test_score_backprop_vs_analytical(self, snapshot_data, prepared_test_data):
        """Test that backprop and analytical score computations match."""
        Y_t = snapshot_data["Y_t"]
        X_t = snapshot_data["X_t"]
        N = prepared_test_data["N"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
        model.distr = 'gamma'

        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        dist_par_un = torch.ones(1)
        beta = torch.zeros(1)

        s_t_ad, _, _ = model.score_t(Y_t, W, beta_t=beta, X_t=X_t, dist_par_un=dist_par_un, backprop_score=True)
        s_t, _, _ = model.score_t(Y_t, W, beta_t=beta, X_t=X_t, dist_par_un=dist_par_un, backprop_score=False)

        # Check that scores are similar
        abs_diff = (s_t - s_t_ad).abs().mean().item()
        assert abs_diff < 0.1, f"Score difference too large: {abs_diff}"


class TestLikelihood:
    """Test likelihood computation."""

    def test_likelihood_types(self, snapshot_data):
        """Test that different likelihood types can be computed without errors."""
        Y_t = snapshot_data["Y_t"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        phi_t = model.start_phi_from_obs(Y_t)

        dist_par_un = torch.ones(1)

        # Test different likelihood types
        logl_type0 = model.loglike_t(Y_t, phi_t, beta=None, X_t=None, dist_par_un=dist_par_un, like_type=0)
        logl_type1 = model.loglike_t(Y_t, phi_t, beta=None, X_t=None, dist_par_un=dist_par_un, like_type=1)
        logl_type2 = model.loglike_t(Y_t, phi_t, beta=None, X_t=None, dist_par_un=dist_par_un, like_type=2)

        # All likelihood values should be finite
        assert torch.isfinite(logl_type0), "Likelihood type 0 should be finite"
        assert torch.isfinite(logl_type1), "Likelihood type 1 should be finite"
        assert torch.isfinite(logl_type2), "Likelihood type 2 should be finite"

    def test_likelihood_components(self, snapshot_data):
        """Test individual likelihood components computation."""
        Y_t = snapshot_data["Y_t"]

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
        phi = model.start_phi_from_obs(Y_t)
        dist_par_un = torch.ones(1)

        A_mat = Y_t > 0
        log_EYcond_mat = model.cond_exp_Y(phi, ret_log=True)

        # Compute likelihood components
        tmp = (dist_par_un - 1) * torch.sum(torch.log(Y_t[A_mat]))
        tmp1 = -torch.sum(A_mat) * torch.lgamma(dist_par_un)
        tmp2 = -dist_par_un * torch.sum(log_EYcond_mat)
        tmp3 = -torch.sum(torch.exp(torch.log(Y_t[A_mat]) - log_EYcond_mat[A_mat]))

        # All components should be finite
        assert torch.isfinite(tmp), "Component tmp should be finite"
        assert torch.isfinite(tmp1), "Component tmp1 should be finite"
        assert torch.isfinite(tmp2), "Component tmp2 should be finite"
        assert torch.isfinite(tmp3), "Component tmp3 should be finite"


class TestScoreDrivenDynamics:
    """Test Score Driven (SD) estimation methods."""

    def test_sd_filter_basic(self, prepared_test_data):
        """Test basic score driven filtering."""
        Y_T = prepared_test_data["Y_T"]
        N = prepared_test_data["N"]
        T = prepared_test_data["T"]

        B = torch.cat([torch.ones(N) * 0.95, torch.ones(N) * 0.95])
        A = torch.cat([torch.ones(N) * 0.01, torch.ones(N) * 0.01])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)

        # Test SD filter
        phi_T, beta_sd_T = model.sd_filt(W, B, A, Y_T)

        assert phi_T is not None, "SD filter should return phi_T"
        assert phi_T.shape[1] == T, f"phi_T should have {T} time steps"

    def test_estimate_sd_basic(self, prepared_test_data):
        """Test basic score driven parameter estimation."""
        Y_T = prepared_test_data["Y_T"]
        N = prepared_test_data["N"]

        B = torch.cat([torch.ones(N) * 0.95, torch.ones(N) * 0.95])
        A = torch.cat([torch.ones(N) * 0.01, torch.ones(N) * 0.01])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
        model.backprop_sd = False

        W_est, B_est, A_est, dist_par_un_est, sd_par_0, diag = model.estimate_SD(
            Y_T,
            B0=B,
            A0=A,
            W0=W,
            sd_par_0=None,
            init_filt_um=False,
            max_opt_iter=20,
            lr=0.01,
            print_every=1,
            dim_dist_par_un=N,
            print_flag=False,
            plot_flag=False,
            est_dis_par_un=False
        )

        assert W_est is not None, "Should estimate W"
        assert B_est is not None, "Should estimate B"
        assert A_est is not None, "Should estimate A"

    def test_estimate_sd_with_static_regressors(self, prepared_test_data):
        """Test score driven estimation with static regressors."""
        Y_T = prepared_test_data["Y_T"]
        X_T_multi = prepared_test_data["X_T_multi"]
        N = prepared_test_data["N"]

        dim_beta = 1
        n_reg = X_T_multi.shape[2]
        n_beta_tv = 0
        dim_dist_par_un = 1

        B = torch.cat([torch.ones(N) * 0.95, torch.ones(N) * 0.95])
        A = torch.cat([torch.ones(N) * 0.01, torch.ones(N) * 0.01])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
        model.distr = 'lognormal'

        W_est, B_est, A_est, dist_par_un_est, beta_const_est, sd_par_0, diag = model.estimate_SD_X0(
            Y_T,
            X_T=X_T_multi,
            dim_beta=dim_beta,
            n_beta_tv=n_beta_tv,
            est_dis_par_un=False,
            dim_dist_par_un=dim_dist_par_un,
            B0=torch.cat((B, torch.zeros(n_beta_tv))),
            A0=torch.cat((A, torch.zeros(n_beta_tv))),
            W0=torch.cat((W, torch.zeros(n_beta_tv))),
            beta_const_0=torch.zeros(dim_beta, n_reg - n_beta_tv),
            max_opt_iter=5,
            lr=0.01,
            print_flag=False,
            plot_flag=False,
            print_every=1
        )

        assert W_est is not None, "Should estimate W"
        assert beta_const_est is not None, "Should estimate beta"

    def test_estimate_sd_with_time_varying_regressors(self, prepared_test_data):
        """Test score driven estimation with time-varying regressors."""
        Y_T = prepared_test_data["Y_T"]
        X_T_multi = prepared_test_data["X_T_multi"]
        N = prepared_test_data["N"]

        dim_beta = 1
        n_reg = X_T_multi.shape[2]
        n_beta_tv = 1
        dim_dist_par_un = N

        B = torch.cat([torch.ones(N) * 0.95, torch.ones(N) * 0.95])
        A = torch.cat([torch.ones(N) * 0.01, torch.ones(N) * 0.01])
        wI, wO = 1 + torch.randn(N), torch.randn(N)
        W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
        model.distr = 'lognormal'

        W_est, B_est, A_est, dist_par_un_est, beta_const_est, sd_par_0, diag = model.estimate_SD_X0(
            Y_T,
            X_T=X_T_multi,
            dim_beta=dim_beta,
            n_beta_tv=n_beta_tv,
            est_dis_par_un=True,
            dim_dist_par_un=dim_dist_par_un,
            B0=torch.cat((B, torch.zeros(n_beta_tv))),
            A0=torch.cat((A, torch.zeros(n_beta_tv))),
            W0=torch.cat((W, torch.zeros(n_beta_tv))),
            beta_const_0=torch.zeros(dim_beta, n_reg - n_beta_tv),
            max_opt_iter=20,
            lr=0.01,
            print_flag=False,
            plot_flag=False
        )

        assert W_est is not None, "Should estimate W"
        assert dist_par_un_est is not None, "Should estimate dist_par_un"
        assert beta_const_est is not None, "Should estimate beta"
