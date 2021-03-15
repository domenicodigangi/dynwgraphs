#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:39:31 2019

dirBin1_Nets :  directew binary model with one parameter per node, aka beta 
                model, directed configuration model, fitness model etc
@author: domenico
"""
import jax.numpy as npj
from jax import grad, hessian, jit
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import numpy as np


from utils import splitVec, putZeroDiag,gen_test_net,degIO_from_mat,strIO_from_mat,npj_t,optim_jax


# Constants:
targetErrValStaNets = 1e-4


def parMat_NetModelDirBin1_(vParUn):
    """ given the vector of unrestricted parameters, the matrix of 
    products of exponentials
    """
    parI,parO = splitVec(vParUn)
    parO = npj_t(parO)
    return  putZeroDiag( npj.multiply(npj.exp(parI) , npj.exp(parO) ) )
parMat_NetModelDirBin1 = jit(parMat_NetModelDirBin1_)


def expMat_NetModelDirBin1_(vParUn):
    """ given the vector of unrestricted parameters, compute the expected 
    matrix
    """
    parMat = parMat_NetModelDirBin1(vParUn)
    return npj.divide( parMat,(1 + parMat))
expMat_NetModelDirBin1 = jit(expMat_NetModelDirBin1_)

def zeroDegParFun(degIO,parIO=None):
    zeroDegPar = -1e6 
    return zeroDegPar,zeroDegPar
    
def phiFunc_(vParUn,ldegI,ldegO):
    """ function that defines one iteration of the estimation map
    of Chatterjee, Diaconis and  Sly for the directed beta model
    """
    vParReI,vParReO = splitVec( npj.exp(vParUn))
    vParReO = npj.reshape(vParReO,[-1,1])
    matI = putZeroDiag( 1/(  vParReI +  (1/vParReO)    )  )
    matO = putZeroDiag( 1/( (1/vParReI) + vParReO ) )
    outPhiI = (ldegI - npj.log(npj.sum(matI,axis = 1)))
    outPhiO = (ldegO - npj.log(npj.sum(matO,0)))
    return npj.concatenate((outPhiI,outPhiO))
phiFunc = jit(phiFunc_)

def logl_NetModelDirBin1(degIO,vParUn):
     tmp1 = npj.sum( npj.multiply(vParUn,degIO))
     tmp2 = npj.sum(npj.log(1 + parMat_NetModelDirBin1(vParUn)) )
     return   tmp1 - tmp2

def estimate_NetModelDirBin1(degIO, targetErr = targetErrValStaNets):
    """ estimate the directer beta model (aka fitness, aka configuration) 
    following the very fast approach of Chatterjee, Diaconis and  Sly 
    """
    degI,degO = splitVec(degIO)
    N = degI.shape[0]
    ldegI = npj.log(degI);ldegO = npj.log(degO)
    #L = npj.sum(degI)
    #LperLink = L/(N**2 - N)
    # unifstartval =  0.5*npj.log((LperLink/((1-LperLink))))
    nnzInds = degIO != 0
    def check_fit_(vParUn,degIO):
        expMat = expMat_NetModelDirBin1(vParUn)
        errsIO = (strIO_from_mat(expMat) - degIO)[nnzInds]
        relErrsIO = npj.divide(npj.abs(errsIO), degIO[nnzInds] )
        return npj.abs(relErrsIO) #check if the constraint is satisfied for all degs    
    check_fit =(check_fit_)
#    if (npj.sum(degI) - npj.sum(degO) ) > 0.1 :
#        raise Exception("sums of in and out degrees should be equal")
    
    maxIt = 100
    it = 0
    vParUn = npj.concatenate((ldegI,ldegO))
    relErrMax = 10
    while (relErrMax > targetErr) & (it<maxIt):
        vParUn = phiFunc(vParUn,ldegI,ldegO)
        relErrMax = npj.max(check_fit(vParUn,degIO))
        it+=1
        #print(it)
        
    zeroParI,zeroParO = zeroDegParFun(degIO)
    
    vParOut = np.array(vParUn)
    vParOut[0:N][degIO[0:N]==0] = zeroParI
    vParOut[N:2*N][degIO[N:2*N]==0] = zeroParO
    vParOut = npj.array(vParUn)
    
    if relErrMax > targetErr:
        print(relErrMax,npj.mean(check_fit(vParUn,degIO)))
        obj_fun_ = lambda vParUn: -logl_NetModelDirBin1(degIO,vParUn)
        vParOut,info_opt = optim_jax(obj_fun_,vParOpt)
    
    return vParOut,it,info_opt



#%% Test estimation of the 
N = 50
density = 0.5

Y_mat = gen_test_net(N,density)
A_mat = npj.greater(Y_mat,0)

degIO = strIO_from_mat(A_mat)
#test = jit(estimate_NetModelDirBin1)
vParOpt,it,info_opt = estimate_NetModelDirBin1(degIO)
 
np.max(np.abs(strIO_from_mat(A_mat) - strIO_from_mat( expMat_NetModelDirBin1(vParOpt)) ))



 