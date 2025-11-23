#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: Static Weighted Network Estimation

This example demonstrates how to estimate parameters for a static weighted
directed network using the Gamma distribution.
"""

import torch
import matplotlib.pyplot as plt
from dynwgraphs.utils.dgps import get_dgp_mod_and_par, sample_T_from_dgp

# Set random seed for reproducibility
torch.manual_seed(42)

print("="*60)
print("Example 1: Static Weighted Network")
print("="*60)

# 1. Generate synthetic network data
print("\n1. Generating synthetic network data...")
N = 20  # Number of nodes
T = 50  # Number of time periods

# Create DGP (Data Generating Process)
dgp_mod, dgp_par = get_dgp_mod_and_par(
    N=N,
    distr='gamma',
    phi_dgp_type='rndwalk',
    size_phi_t='2N',
    phi_tv=False,  # Static parameters
    dist_par_dgp_type='uniform',
    size_dist_par_un_t='N'
)

# Sample network observations
Y_T = sample_T_from_dgp(dgp_mod, dgp_par, T=T, burn_in=10)

print(f"Generated network: {N} nodes, {T} time periods")
print(f"Data shape: {Y_T.shape}")
print(f"Sparsity: {(Y_T == 0).float().mean():.2%} zero edges")

# 2. Estimate model parameters
print("\n2. Estimating model parameters...")

# Import model class
from dynwgraphs.dirGraphs1_dynNets import dirSpW1_sequence_ss

# Initialize model
model = dirSpW1_sequence_ss()
model.distr = 'gamma'

# Initialize parameters
model.init_all_par_sequences(Y_T, T_train=T)

# Get starting values
phi_T_0 = model.start_phi_from_obs_T(Y_T)
dist_par_un_0 = model.dist_par_un_start_val(dim=N)

# Estimate parameters for each time period
print("Running estimation (this may take a moment)...")
phi_T_est, diagnostics = model.ss_filt(
    Y_T,
    phi_T_0=phi_T_0,
    dist_par_un=dist_par_un_0,
    est_dist_par=False,  # Keep distribution parameters fixed
    max_opt_iter=50,
    opt_n='ADAM',
    lr=0.01,
    print_flag=False
)

print("Estimation complete!")

# 3. Evaluate fit
print("\n3. Evaluating model fit...")

# Get expected values under estimated parameters
t_example = 25  # Example time period
Y_t = Y_T[:, :, t_example]
phi_t = phi_T_est[:, t_example]

# Compute expected adjacency matrix
E_Y_t = model.cond_exp_Y(phi_t)

# Compute metrics on non-zero edges only
nonzero_mask = Y_t > 0
if nonzero_mask.sum() > 0:
    obs = Y_t[nonzero_mask]
    pred = E_Y_t[nonzero_mask]
    relative_error = torch.abs(obs - pred) / obs
    print(f"\nTime period {t_example}:")
    print(f"  Number of non-zero edges: {nonzero_mask.sum()}")
    print(f"  Mean relative error: {relative_error.mean():.3f}")
    print(f"  Median relative error: {relative_error.median():.3f}")

# 4. Visualization
print("\n4. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Network density over time
density_T = (Y_T > 0).float().mean(dim=(0, 1))
axes[0, 0].plot(density_T.numpy())
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Network Density')
axes[0, 0].set_title('Network Density Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Observed vs Expected (sample time period)
if nonzero_mask.sum() > 10:
    sample_indices = torch.randperm(nonzero_mask.sum())[:100]
    axes[0, 1].scatter(obs[sample_indices], pred[sample_indices], alpha=0.5)
    axes[0, 1].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'r--', label='Perfect fit')
    axes[0, 1].set_xlabel('Observed Edge Weight')
    axes[0, 1].set_ylabel('Expected Edge Weight')
    axes[0, 1].set_title(f'Fit Quality (t={t_example})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Parameter evolution (in-strength parameters)
phi_in = phi_T_est[:N, :]
for i in range(min(5, N)):  # Plot first 5 nodes
    axes[1, 0].plot(phi_in[i, :].detach().numpy(), alpha=0.7, label=f'Node {i}')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('φ (in-strength parameter)')
axes[1, 0].set_title('Parameter Evolution (In-Strength)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Adjacency matrix heatmap
im = axes[1, 1].imshow(Y_t.numpy(), cmap='YlOrRd', aspect='auto')
axes[1, 1].set_xlabel('Target Node')
axes[1, 1].set_ylabel('Source Node')
axes[1, 1].set_title(f'Adjacency Matrix (t={t_example})')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('static_weighted_network_example.png', dpi=150)
print("Saved figure: static_weighted_network_example.png")

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
