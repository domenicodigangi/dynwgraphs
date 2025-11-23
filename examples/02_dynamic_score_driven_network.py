#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: Dynamic Score-Driven Network

This example demonstrates how to estimate time-varying parameters using
score-driven dynamics for a weighted directed network.
"""

import torch
import matplotlib.pyplot as plt
from dynwgraphs.dirGraphs1_dynNets import dirSpW1_SD
from dynwgraphs.utils.dgps import get_dgp_mod_and_par, sample_T_from_dgp

# Set random seed
torch.manual_seed(123)

print("="*60)
print("Example 2: Dynamic Score-Driven Network")
print("="*60)

# 1. Generate synthetic dynamic network data
print("\n1. Generating synthetic dynamic network data...")
N = 15  # Number of nodes
T = 80  # Number of time periods
T_train = 60  # Training period
T_test = T - T_train  # Test period

# Create DGP with time-varying parameters
dgp_mod, dgp_par = get_dgp_mod_and_par(
    N=N,
    distr='gamma',
    phi_dgp_type='rndwalk',
    size_phi_t='2N',
    phi_tv=True,  # Time-varying parameters
    dist_par_dgp_type='uniform',
    size_dist_par_un_t='N'
)

# Sample observations
Y_T = sample_T_from_dgp(dgp_mod, dgp_par, T=T, burn_in=10)

print(f"Generated network: {N} nodes, {T} time periods")
print(f"Training period: {T_train}, Test period: {T_test}")
print(f"Sparsity: {(Y_T == 0).float().mean():.2%} zero edges")

# 2. Estimate score-driven model
print("\n2. Estimating score-driven dynamics...")

# Initialize model
model = dirSpW1_SD()
model.distr = 'gamma'

# Set up model
model.init_all_par_sequences(Y_T, T_train=T_train)

# Estimate all parameters (phi dynamics + static dist_par)
print("Running score-driven estimation...")
sd_par_est, diagnostics = model.estimate_SD(
    Y_T[:, :, :T_train],
    max_opt_iter=30,
    opt_n='ADAM',
    lr=0.01,
    print_flag=True,
    print_every=5
)

print("\nEstimation complete!")

# 3. Filtering and forecasting
print("\n3. Performing filtering and forecasting...")

# Filter through training period
phi_T_filt, _ = model.sd_filt(
    Y_T[:, :, :T_train],
    par_list=sd_par_est
)

# Forecast out-of-sample
print(f"Forecasting {T_test} steps ahead...")
forecasts = []
for t in range(T_train, T):
    # One-step-ahead forecast
    phi_forecast = model.get_phi_SD_one_step_ahead_forecast(
        Y_T[:, :, :t],
        par_list=sd_par_est,
        t_since_init_SD=t-1
    )
    E_Y_forecast = model.cond_exp_Y(phi_forecast)
    forecasts.append(E_Y_forecast)

# 4. Evaluate forecast performance
print("\n4. Evaluating forecast performance...")

# Compute metrics
mae_list = []
for t_idx, t in enumerate(range(T_train, T)):
    Y_true = Y_T[:, :, t]
    Y_pred = forecasts[t_idx]

    # Only evaluate on non-zero edges
    nonzero = Y_true > 0
    if nonzero.sum() > 0:
        mae = torch.abs(Y_true[nonzero] - Y_pred[nonzero]).mean()
        mae_list.append(mae.item())

if mae_list:
    print(f"Mean Absolute Error (out-of-sample): {sum(mae_list)/len(mae_list):.4f}")
    print(f"MAE std: {torch.tensor(mae_list).std():.4f}")

# 5. Visualization
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Parameter dynamics for selected nodes
phi_filt_in = phi_T_filt[:N, :]
for i in range(min(3, N)):
    axes[0, 0].plot(phi_filt_in[i, :].detach().numpy(), label=f'Node {i}')
axes[0, 0].axvline(x=T_train, color='r', linestyle='--', label='Train/Test split')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('φ (in-strength parameter)')
axes[0, 0].set_title('Parameter Dynamics (Score-Driven)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Out-of-sample forecast errors
if mae_list:
    axes[0, 1].plot(range(T_train, T), mae_list, marker='o')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Forecast Error Over Time')
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Total network volume over time
volume_T = Y_T.sum(dim=(0, 1))
axes[1, 0].plot(volume_T.numpy(), label='Observed')
axes[1, 0].axvline(x=T_train, color='r', linestyle='--', label='Train/Test split')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Total Network Volume')
axes[1, 0].set_title('Network Activity')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Forecast vs Observed (last forecast step)
if len(forecasts) > 0:
    Y_last_true = Y_T[:, :, -1]
    Y_last_pred = forecasts[-1]
    nonzero_last = Y_last_true > 0

    if nonzero_last.sum() > 10:
        obs_vals = Y_last_true[nonzero_last].numpy()
        pred_vals = Y_last_pred[nonzero_last].detach().numpy()

        axes[1, 1].scatter(obs_vals, pred_vals, alpha=0.5)
        max_val = max(obs_vals.max(), pred_vals.max())
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect forecast')
        axes[1, 1].set_xlabel('Observed Edge Weight')
        axes[1, 1].set_ylabel('Forecasted Edge Weight')
        axes[1, 1].set_title(f'Forecast Quality (t={T})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dynamic_score_driven_network_example.png', dpi=150)
print("Saved figure: dynamic_score_driven_network_example.png")

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
print("\nKey takeaways:")
print("- Score-driven dynamics allow parameters to evolve over time")
print("- The model can track changes in network structure")
print("- Out-of-sample forecasting is supported")
print("- Evaluation should focus on non-zero edges for sparse networks")
