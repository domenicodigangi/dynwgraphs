#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:17:21 2019

@author: domenico
"""


import jax.numpy as npj
from jax import grad, hessian, jit
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import scipy as sp
import numpy as np

from matplotlib import pyplot as plt

def hist_and_norm(data):
    mu, std = sp.stats.norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = sp.stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()




def splitVec(vec,N=None):
    if N is None:
        N = vec.shape[0]//2
    return vec[:N],vec[N:2*N]


def gen_test_net(N,density):
    Y_mat = np.random.rand(N,N)
    Y_mat[Y_mat > density] = 0    
    Y_mat *= 10000
    Y_mat = npj.array(Y_mat)
    return Y_mat

def degIO_from_mat(Y):
    A = npj.greater(Y,0)
    return npj.concatenate((npj.sum(A,axis=0),npj.sum(A,axis=1)))
def strIO_from_mat(Y):
    return npj.concatenate((npj.sum(Y,axis=0),npj.sum(Y,axis=1)))


def npj_t_(vec):
    """ transpose a jax.numpy 1D vector
    """
    return  npj.reshape(vec,[-1,1])
npj_t = jit(npj_t_)

 
def putZeroDiag_(mat):     
    """ set to zero the diagonal elements of a matrix. 
    written for JAX. the short option 
     mat = jax.ops.index_update(mat, npj.diag_indices(mat.shape[0]),0)
      does not work!
    """
    diag_inds = npj.diag_indices(mat.shape[0])
    mat = jax.ops.index_add(mat,diag_inds , -mat[diag_inds] )
    return mat
putZeroDiag = (putZeroDiag_)



def optim_jax(obj_fun_,x0,print_info = False,jac=None,hess=None):
    all_x = []
    all_j = []
    all_f = []
    def store(x):
            all_x.append(x)
            all_j.append(npj.linalg.norm(jac(x)))
            all_f.append(obj_fun(x))
            print(x,npj.linalg.norm(jac(x)),npj.max(jac(x)),obj_fun(x))
    print("start optimization")
    if jac is None:
        obj_fun = jit(obj_fun_)
        jac = jit(grad(obj_fun))    
        hess = jit(hessian(obj_fun))
        res1 = sp.optimize.minimize(obj_fun, x0 ,callback = store, method = 'trust-ncg', 
        	       options={'disp': print_info}, jac = jac,hess=hess)
    elif jac == -1 :
        obj_fun = obj_fun_
        jac=None
        hess=None
        res1 = sp.optimize.minimize(obj_fun, x0 ,callback = store, method = 'BFGS', 
        	       options={'disp': print_info}, jac = jac,hess=hess)
    else:
        obj_fun = obj_fun_
        res1 = sp.optimize.minimize(obj_fun, x0 ,callback = store, method = 'trust-ncg', 
        	       options={'disp': print_info}, jac = jac,hess=hess)

    return res1.x, (all_x,all_f,all_j)





#%%
import time

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






























#