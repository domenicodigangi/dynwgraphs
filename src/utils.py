"""
functions needed in different parts of the module
"""

import torch
from hypergrad import SGDHD, AdamHD
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import pandas as pd

from lbfgsnew import LBFGSNew

def splitVec(vec, N=None):
    if N is None:
        N = vec.shape[0]//2
    if vec.dim() == 1:
        return vec[:N], vec[N:2*N]
    elif vec.dim() == 2:
        return vec[:N, :], vec[N:2*N, :]

def tens(x, dtype = torch.float32):
    return torch.tensor(x, dtype=dtype)

def gen_test_net(N, density):
    Y_mat = torch.rand(N,N)
    Y_mat[Y_mat > density] = 0
    Y_mat *= 10000
    return Y_mat

def degIO_from_mat(Y):
    A =  ( Y > 0).int()
    return torch.cat((torch.sum(A,axis=0),torch.sum(A,axis=1)))

def strIO_from_mat(Y):
    return torch.cat((torch.sum(Y, axis=0), torch.sum(Y, axis=1)))


def putZeroDiag(mat, diff=True):
    n = mat.shape[0]
    mask = torch.eye(n, n).bool()

    if diff:
        out = (~torch.eye(n, n).bool()).float() * mat
    else:
        out = mat.masked_fill_(mask, 0)
    return out

def putZeroDiag_T(mat_T):
    n = mat_T.shape[0]
    T = mat_T.shape[2]
    mask = torch.eye(n, n).bool()
    mask_T = mask.unsqueeze(2).repeat_interleave(T, dim=2)
    mat_T.masked_fill_(mask_T, 0)
    return mat_T

def soft_lu_bound(x, l_limit=-50, u_limit=40):
    """soft lower and upper bound: force argument into a range """
    m = torch.nn.Tanh()
    dp = (l_limit + u_limit) / 2
    dm = (l_limit - u_limit) / 2
    x = (x - dp) / dm
    return dm * m(x) - dm * m(-(dp / dm) * torch.ones(1))

def soft_l_bound(x, l_limit):
    """ soft lower bound: force  argument into a range bounded from below"""
    error # function not working as intended. to be checked!
    if l_limit>0:
        raise
    m = torch.nn.Softplus()
    x = (x - l_limit)
    return m(x) - m(- l_limit * torch.ones(1))

def rand_steps(start_val, end_val, Nsteps, T, rand=True):
    out = np.zeros(T)
    if Nsteps > 1:
        heights = np.linspace(start_val, end_val, Nsteps)
        if rand:
            np.random.shuffle(heights)

        Tstep = T // Nsteps
        last = 0
        for i in range(Nsteps - 1):
            out[last:last + Tstep] = heights[i]
        last += Tstep
        out[last:] = heights[-1]

    elif Nsteps == 1:
        out = start_val

    return tens(out)

def dgpAR(mu, B, sigma, T, N=1, minMax=None, scaling = "uniform"):

    w = mu * (1-B)
    path = torch.randn(N, T) * sigma
    path[:, 0] = mu

    for t in range(1, T):
        path[:, t] = w + B*path[:, t-1] + path[:, t]

    if minMax is not None:
        min = minMax.min(dim=1)
        max = minMax.max(dim=1)
        if scaling == "uniform":
            minPath = path.min(dim=1)
            maxPath = path.max(dim=1)
            Δ = (max-min)/(maxPath - minPath)
            rescPath = min + (path - minPath)*Δ
        elif scaling == "nonlinear":
              rescPath = min + (max - min) * 1/(1 + torch.exp(path))

    else:
        rescPath = path
    return rescPath

def store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag, print_every,
                             roll_rel_im, no_improv_count, print_par, print_fun=None):

    grad_mean = unPar.grad.abs().mean().item()
    num_par = unPar.shape[0]
    #store info and Print them when required
    if print_fun is not None:
        fun_val = print_fun(unPar.clone().detach())
        diag.append((loss.item(), fun_val.item(), grad_norm, grad_mean, roll_rel_im))
    else:
        diag.append((loss.item(), grad_norm, grad_mean, roll_rel_im))
    tmp = unPar.data
    if i//print_every > last_print_it:
        last_print_it = i//print_every
        if print_fun is not None:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lRate, fun_val))
        elif print_par:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lRate, tmp))
        else:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                    num_par, lRate))
    return last_print_it


def optim_torch(obj_fun_, unPar0, opt_steps=1000, opt_n=1, lRate=0.01, rel_improv_tol=5e-8, no_improv_max_count=50,
                min_n_iter=250, bandwidth=250, small_grad_th=1e-6,
                print_flag=False, print_every=10, print_par=False, print_fun=None, plot_flag=False):
    """given a function and a starting vector, run one of different pox optimizations"""
    unPar = unPar0.clone().detach()
    unPar.requires_grad = True

    optimizers = {"SGD" : torch.optim.SGD([unPar], lr=lRate, nesterov=False),
                  "ADAM": torch.optim.Adam([unPar], lr=lRate),
                  "SGDHD": SGDHD([unPar], lr=lRate, hypergrad_lr=1e-8),
                  "ADAMHD": AdamHD([unPar], lr=lRate, hypergrad_lr=1e-8),
                  "NESTEROV_1" : torch.optim.SGD([unPar], lr=lRate, momentum=0.5, nesterov=True),
                  "NESTEROV_2" : torch.optim.SGD([unPar], lr=lRate, momentum=0.7, nesterov=True),
                  "LBFGS" : torch.optim.LBFGS([unPar], lr=lRate),
                  "LBFGS_NEW" : LBFGSNew([unPar], lr=lRate, line_search_fn = True)}

    if type(opt_n) == int:
        optimizer = list(optimizers.values())[opt_n]
    elif type(opt_n) == str:
        optimizer = optimizers[opt_n]


    def closure():
        # if torch.is_grad_enabled():
        optimizer.zero_grad()
        loss = obj_fun_(unPar)
        # if loss.requires_grad:
        loss.backward()
        return loss
    last_print_it=0
    diag = []
    rel_im = np.ones(0)
    i = 0
    loss = closure()
    last_loss = loss.item()
    no_improv_flag = False
    small_grad_flag = False
    nan_flag = False
    no_improv_count = 0
    while (i <= opt_steps) and (not no_improv_flag) & (not small_grad_flag) & (not nan_flag):
        loss = optimizer.step(closure)

        # check the gradient's norm
        grad_norm = unPar.grad.norm().item()
        small_grad_flag = grad_norm < small_grad_th
        # check presence of nans in opt vector
        nan_flag = torch.isnan(unPar).any().item()
        # check improvement
        #print((i, loss.item()))
        rel_improv = (last_loss - loss.item())
        if not (loss.item() == 0):
            rel_improv = rel_improv/loss.abs().item()
        rel_im = np.append(rel_im, rel_improv)
        last_loss = loss.item()
        if i > min_n_iter:
            roll_rel_im = rel_im[-bandwidth:].mean()
            if roll_rel_im < rel_improv_tol:
                no_improv_count = no_improv_count + 1
            else:
                no_improv_count = 0
            if no_improv_count > no_improv_max_count:
                no_improv_flag = True
        else:
            roll_rel_im = rel_im.mean()

        last_print_it = store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag,
                                                print_every, roll_rel_im, no_improv_count, print_par, print_fun=print_fun)
        i = i+1
    loss = closure()
    grad_norm = unPar.grad.norm().item()
    store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag,
                             print_every, -99999.0, no_improv_count, print_par, print_fun=print_fun)

    if plot_flag:
        plt.figure()
        plt.plot(diag)
        plt.legend(legend[opt_n])

    unPar_est = unPar.clone()
    return unPar_est, diag


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def likel_ratio_p_val(logl_0, logl_1, df):
    G = 2 * (logl_1 - logl_0)
    p_value = chi2.sf(G, df)
    return p_value


def req_grads(parTuple):
    for p in parTuple:
        p.requires_grad = True
    # # sd par 0 does not require grad
    # parTuple[-1].requires_grad = False

def zero_grads(parTuple):
    for p in parTuple:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()























#