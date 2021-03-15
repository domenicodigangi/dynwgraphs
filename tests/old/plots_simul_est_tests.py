

#


import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("./src/")

def eval_est(err_vec):
    # given err_vec compute the stats needed to evaluate the estimate
    return err_vec.mean(), np.square(err_vec).mean(), np.abs(err_vec).mean()

#%% ----------------------------------------- Increasing N steps ----------------------------

list = [[250, 200], [500, 100], [1000, 50], [1500, 50]]

res = np.zeros((4, 3, 2, len(list)))

for ind_list in range(len(list)):
    N_steps, N_sample = list[ind_list]
    N = 20
    learn_rate = 0.005
    T = 500

    dim_beta = (1)
    LOAD_FOLD = './data/estimates_test'
    file_path = LOAD_FOLD + '/dirSpW1_X0_dynNet_SD_est_test_lr_' + \
                str(learn_rate) + '_T_' + str(T) + '_N_steps_' + \
                str(N_steps) + '_dim_beta_' + str(dim_beta) + \
                '_Nsample_' + str(N_sample) + '.npz'

    tmp = np.load(file_path)
    all_W, all_B, all_A, all_beta, w, B, A, beta = \
        tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5'], tmp['arr_6'], tmp['arr_7']

    plot_flag = False
    if plot_flag:
        plt.close('all')
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].set_title('N Steps = ' + str(N_steps))

    for ind_dens in [0, 1]:
        for ind_B in range(B.shape[0]):
            err = all_B[ind_B, :, ind_dens].transpose() - B[ind_B]
            res[0, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[0, ind_dens].hist(err, alpha=0.7)

        for ind_A in range(A.shape[0]):
            err = all_A[ind_A, :, ind_dens].transpose() - A[ind_A]
            res[1, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[1, ind_dens].hist(err, alpha=0.7)

        err = all_beta[0, :, ind_dens].transpose() - beta
        res[2, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[2, ind_dens].hist(err, alpha=0.7)
        err = (all_W[:, :, ind_dens] - w.reshape((-1, 1))).transpose()
        res[3, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[3, ind_dens].hist(err.flatten())

plt.close('all')
fig, ax = plt.subplots(4)
ax[0].set_title('N  = ' + str(N_steps))
x_lab = [_[0] for _ in list]
for ind_par in range(4):
    ax[ind_par].plot(x_lab, res[ind_par, 1, :, :].transpose())



#%%----------------------------------------- Change Time lenght ----------------------------

list = [2500, 5000]


res = np.zeros((4, 3, 2, len(list)))

for ind_list in range(len(list)):
    N_steps = list[ind_list]
    N_sample = 200
    N = 20
    learn_rate = 0.005
    T = 50

    dim_beta = (1)
    LOAD_FOLD = './data/estimates_test'
    file_path = LOAD_FOLD + '/dirSpW1_X0_dynNet_SD_est_test_lr_' + \
                str(learn_rate) + '_T_' + str(T) + '_N_steps_' + \
                str(N_steps) + '_dim_beta_' + str(dim_beta) + \
                '_Nsample_' + str(N_sample) + '.npz'

    tmp = np.load(file_path)
    all_W, all_B, all_A, all_beta, w, B, A, beta = \
        tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5'], tmp['arr_6'], tmp['arr_7']

    plot_flag = False
    if plot_flag:
        plt.close('all')
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].set_title('N Steps = ' + str(N_steps))

    for ind_dens in [0, 1]:
        for ind_B in range(B.shape[0]):
            err = all_B[ind_B, :, ind_dens].transpose() - B[ind_B]
            res[0, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[0, ind_dens].hist(err, alpha=0.7)

        for ind_A in range(A.shape[0]):
            err = all_A[ind_A, :, ind_dens].transpose() - A[ind_A]
            res[1, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[1, ind_dens].hist(err, alpha=0.7)

        err = all_beta[0, :, ind_dens].transpose() - beta
        res[2, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[2, ind_dens].hist(err, alpha=0.7)
        err = (all_W[:, :, ind_dens] - w.reshape((-1, 1))).transpose()
        res[3, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[3, ind_dens].hist(err.flatten())

plt.close('all')
fig, ax = plt.subplots(4)
ax[0].set_title('N  = ' + str(N_steps))
x_lab = [_ for _ in list]
for ind_par in range(4):
    ax[ind_par].plot(x_lab, res[ind_par, 0, :, :].transpose())




##


#%% ----------------------------------------- Changing learning rate ----------------------------

list = [0.001, 0.01]# [0.05, 0.025, 0.01, 0.0075, 0.005, 0.001, 0.0005]
N_steps = 10000
N_sample = 150
N = 20
T = 50

res = np.zeros((4, 3, 2, len(list)))
ind_list = 0
ind_dens = 0
ind_B = 10
for ind_list in range(len(list)):
    learn_rate = list[ind_list]
    dim_beta = (1)
    LOAD_FOLD = './data/estimates_test'
    file_path = LOAD_FOLD + '/dirSpW1_X0_dynNet_SD_est_test_lr_' + \
                str(learn_rate) + '_T_' + str(T) + '_N_steps_' + \
                str(N_steps) + '_dim_beta_' + str(dim_beta) + \
                '_Nsample_' + str(N_sample) + '_N_SDpars.npz'

    tmp = np.load(file_path)
    all_W, all_B, all_A, all_beta, w, B, A, beta = \
        tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5'], tmp['arr_6'], tmp['arr_7']

    plot_flag = False
    if plot_flag:
        plt.close('all')
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].set_title('N Steps = ' + str(N_steps))

    for ind_dens in [0, 1]:
        for ind_B in range(B.shape[0]):
            err = all_B[ind_B, :, ind_dens].transpose() - B[ind_B]
            res[0, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[0, ind_dens].hist(err, alpha=0.7)

        for ind_A in range(A.shape[0]):
            err = all_A[ind_A, :, ind_dens].transpose() - A[ind_A]
            res[1, :, ind_dens, ind_list] = eval_est(err)
            if plot_flag:
                ax[1, ind_dens].hist(err, alpha=0.7)

        err = all_beta[0, :, ind_dens].transpose() - beta
        res[2, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[2, ind_dens].hist(err, alpha=0.7)
        err = (all_W[:, :, ind_dens] - w.reshape((-1, 1))).transpose()
        res[3, :, ind_dens, ind_list] = eval_est(err)
        if plot_flag:
            ax[3, ind_dens].hist(err.flatten())

#%%
plt.close('all')
fig, ax = plt.subplots(4, 2)
ax[0,0].set_title('N  = ' + str(N_steps))
x_lab = [_ for _ in list]
leg_par = ["B", "A", "beta", "W"]
for ind_measure in range(2):
    for ind_par in range(4):
        ax[ind_par, ind_measure].plot(x_lab, res[ind_par, ind_measure, :, :].transpose(), '-*')
        ax[ind_par, ind_measure].set_title(leg_par[ind_par])
        ax[ind_par, ind_measure].legend(["Dens = 0.2", "Dens = 0.8"])

































    #