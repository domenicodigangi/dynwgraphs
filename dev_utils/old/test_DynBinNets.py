
import torch
import numpy as np
import sys
sys.path.append("./src/")
from utils import splitVec, putZeroDiag,gen_test_net,degIO_from_mat,strIO_from_mat,tic,toc

from dirBin1_dynNets import dirBin1_staNet,  dirBin1_X0_dynNet_SD

#%% Test static model estimate
N = 50
density = 0.5

Y_mat = gen_test_net(N, density)
A_mat = (Y_mat > 0).float()

model = dirBin1_staNet()
vParOpt, it, err_list = model.estimateBin(A_mat, plot_flag=True)
degIO = degIO_from_mat(A_mat)

torch.max(model.check_fitBin(vParOpt, degIO))

# test dynamical version




# test many estimates
N = 20
T = 50
learn_rate = 0.001

N_steps = 10000
N_sample = 150

N_BA = N
p_list = [0.2, 0.8]
dim_beta = (N)

all_W = np.zeros((2*N, N_sample, len(p_list)))
all_B = np.zeros((2*N_BA, N_sample, len(p_list)))
all_A = np.zeros((2*N_BA, N_sample, len(p_list)))
all_beta = np.zeros((dim_beta, N_sample, len(p_list)))

# Define DGP parameters -----------------------------------------------------
beta = 0.1*torch.ones(dim_beta)
B = torch.cat([torch.randn(N_BA)*0.03 + 0.9, torch.randn(N_BA)*0.03 + 0.8])
A = torch.cat([torch.randn(N_BA)*0.003 + 0.02, torch.randn(N_BA)*0.003 + 0.02])
wI, wO = 1 + torch.randn(N), torch.randn(N)
w = torch.cat((torch.ones(N)*wI, torch.ones(N)*wO)) * 0.001
alpha = torch.ones(1)
# generate random regressors
X_T = torch.randn(N, N, T)

# starting points for the optimization-------------------------------------------
B0 = torch.cat([torch.ones(N_BA) * 0.7, torch.ones(N_BA) * 0.7])
A0 = torch.cat([torch.ones(N_BA) * 0.0001, torch.ones(N_BA) * 0.0001])
wI, wO = 3 + torch.randn(N), torch.randn(N)
W0 = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001


#%%
model = dirBin1_X0_dynNet_SD()

w = model.identify(w)
W0 = model.identify(W0)
theta_T, A_T = model.sd_dgp_bin(w, B, A, T, beta=beta, X_T=X_T)

#plt.plot(theta_T.detach().numpy().transpose())

#%%
W_est, B_est, A_est, beta_est, diag = model.estimate(A_T, X_T, B0=B0, A0=A0, W0=W0,
                                                     dim_beta=dim_beta, Steps=3,
                                                     lr=learn_rate)








