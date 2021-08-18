#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 16th 2021

"""


"""
functions needed in different parts of the module
"""

import torch

def splitVec(vec, N=None):
    if N is None:
        N = vec.shape[0]//2
    if vec.dim() == 1:
        return vec[:N], vec[N:2*N]
    elif vec.dim() == 2:
        return vec[:N, :], vec[N:2*N, :]

def tens(x, dtype = torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype)
    else:
        return torch.tensor(x, dtype=dtype)

def degIO_from_mat(Y):
    A =  ( Y > 0).int()
    return torch.cat((torch.sum(A,axis=0),torch.sum(A,axis=1)))

def strIO_from_mat(Y):
    return torch.cat((torch.sum(Y, axis=0), torch.sum(Y, axis=1)))

def strIO_from_tens_T(Y_T):
    return torch.cat((torch.sum(Y_T, dim=(0,2)), torch.sum(Y_T, dim=(1,2))))

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

def soft_lu_bound(x, l_limit, u_limit):
    """soft lower and upper bound: force argument into a range """
    m = lambda x: torch.tanh(x)
    dp = (l_limit + u_limit) / 2
    dm = (l_limit - u_limit) / 2
    y = dm * m( (x - dp) / dm ) - dm * m(-(dp / dm) * torch.ones(1))
    return y

def inv_soft_lu_bound(y, l_limit, u_limit):
    """inverse soft lower and upper bound"""
    m = lambda x: torch.tanh(x)
    m_inv = lambda x: torch.atanh(x)
    dp = (l_limit + u_limit) / 2
    dm = (l_limit - u_limit) / 2
    x = m_inv( (y / dm) + m(-(dp / dm)* torch.ones(1) ) ) * dm + dp
    return x
    
def soft_l_bound(x, l_limit):
    """ soft lower bound: force  argument into a range bounded from below"""
    error # function not working as intended. to be checked!
    if l_limit>0:
        raise
    m = torch.nn.Softplus()
    x = (x - l_limit)
    return m(x) - m(- l_limit * torch.ones(1))


def size_from_str(pox_str, N):
        if type(pox_str) == int:
            return pox_str
        elif pox_str == "one":
            return 1
        elif pox_str == "N":
            return N
        elif pox_str == "2N":
            return N * 2


#