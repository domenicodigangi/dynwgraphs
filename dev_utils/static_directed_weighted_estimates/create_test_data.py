#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday June 11th 2021

"""


# %% import packages
import numpy as np
import matplotlib.pyplot as plt
import mlflow

# %%
experiment_name = "static directed weighted estimates"
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


# %% 
test_data = np.load("../tests/test_data/dir_w_test_data.npz")
# define input data for tests 
unit_measure = 1e5
Y_T = tens(test_data["Y_T"])/unit_measure
X_scalar_T = tens(test_data["X_T"]).unsqueeze(1).unsqueeze(1).permute((3,2,1,0))
X_matrix_T = Y_T[:, :, :-1].log().unsqueeze(2)
X_matrix_T[~torch.isfinite(X_matrix_T)] = 0
N = Y_T.shape[0]
T = Y_T.shape[2]
t = T - 12
Y_t = Y_T[:, :, t]
X_t = X_matrix_T[:, :, :, t]
A_t = Y_t > 0
Y_tp1 = Y_T[:, :, t+1]

X_T_multi = X_matrix_T.repeat_interleave(2, dim=2)
X_T_multi[:, :, 1, :] += 1

# %%
import click

@click.command()
@click.option("--n-sample", default=0.8, type=float)
@click.option("--max-opt-iter", default=5000, type=int)
@click.option("--ovflw-lm", default=True, type=bool)
@click.option("--distr", default="gamma", type=str)

def sample_and_estimate(n_sample, max_opt_iter, avoid_ovflw_fun_flag, distr):

    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distr = distr)

    mlflow.log_metric("N", Y_t.shape[0])

    print("The model had a MSE on the test set of {0}".format(test_mse))
    print("The model had a MSE on the (train) set of {0}".format(train_mse))
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_mse", train_mse)


if __name__ == "__main__":
    sample_and_estimate()

