#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:39:31 2019


@author: domenico
"""

import torch
torch.set_printoptions(precision=10)
dtype = torch.float64
device = torch.device("cpu")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from src.utils import splitVec, putZeroDiag,gen_test_net,degIO_from_mat,strIO_from_mat

# Constants:
targetErrValStaNets = 1e-4

#%%

def parMat_NetModelDirBin1(vParUn):
    """ given the vector of unrestricted parameters, the matrix of 
    products of exponentials
    """
    parI,parO = splitVec(vParUn)
    return  putZeroDiag( torch.mul(torch.exp(parI) , torch.exp(parO).unsqueeze(1) ) )

def expMat_NetModelDirBin1(vParUn):
    """ given the vector of unrestricted parameters, compute the expected 
    matrix
    """
    parMat = parMat_NetModelDirBin1(vParUn)
    return torch.div( parMat,(1 + parMat))

def zeroDegParFun(degIO,parIO=None):
    zeroDegPar = -1e6 
    return zeroDegPar,zeroDegPar
    
def phiFunc(vParUn,ldegI,ldegO):
    """ function that defines one iteration of the estimation map
    of Chatterjee, Diaconis and  Sly for the directed beta model
    """
    vParReI,vParReO = splitVec( torch.exp(vParUn))
    vParReO = vParReO.unsqueeze(1)
    matI = putZeroDiag( 1/(  vParReI +  (1/vParReO)    )  )
    matO = putZeroDiag( 1/( (1/vParReI) + vParReO ) )
    outPhiI = (ldegI - torch.log(torch.sum(matI,axis = 1)))
    outPhiO = (ldegO - torch.log(torch.sum(matO,0)))
    return torch.cat((outPhiI,outPhiO))

def logl_NetModelDirBin1(degIO,vParUn):
     tmp1 = torch.sum( torch.mul(vParUn,degIO))
     tmp2 = torch.sum(torch.log(1 + parMat_NetModelDirBin1(vParUn)) )
     return   tmp1 - tmp2

def estimate_NetModelDirBin1_MLE(degIO, unPar0, Steps =50, opt_n = 1, plot_flag = False):
    """Estimate model's parameters by optimizing the loglikelihood"""
    unPar = unPar0.clone()
    unPar.requires_grad_(True)
    obj_fun = lambda :  -logl_NetModelDirBin1(degIO, unPar)
    # def obj_fun(vParUn):
    #    return  -logl_NetModelDirBin1(degIO, vParUn)

    optimizers = [torch.optim.SGD([unPar], lr=1e-5, nesterov=False),
                  torch.optim.Adam([unPar], lr=0.01),
                  torch.optim.SGD([unPar], lr=0.0001, momentum=0.5, nesterov=True),
                  torch.optim.SGD([unPar], lr=0.0001, momentum=0.7, nesterov=True)]
    legend = ["SGD", "Adam", "Nesterov 1", "Nesterov 2"]
    diag = []
    for i in range(Steps):
        loss = obj_fun()
        # set all gradients to zero
        optimizers[opt_n].zero_grad()
        # compute the gradients
        loss.backward(retain_graph=True)
        # take a step
        optimizers[opt_n].step()
        #print((loss.data, unPar.data))
        diag.append(loss.item())
    par_est = unPar.clone()
    if plot_flag:
        plt.figure()
        plt.plot(diag)
        plt.legend(legend[opt_n])

    return par_est

def check_fit(vParUn,degIO,nnzInds = None):
    if nnzInds is None:
        nnzInds = degIO != 0
    expMat = expMat_NetModelDirBin1(vParUn)
    errsIO = (strIO_from_mat(expMat) - degIO)[nnzInds]
    relErrsIO = torch.div(torch.abs(errsIO), degIO[nnzInds] )
    return torch.abs(relErrsIO) #check if the constraint is satisfied for all degs

def estimate_NetModelDirBin1(degIO, targetErr = targetErrValStaNets,plot_flag = False):
    """ estimate the directer beta model (aka fitness, aka configuration) 
    following the very fast approach of Chatterjee, Diaconis and  Sly 
    """
    degI,degO = splitVec(degIO)
    N = degI.shape[0]
    ldegI = torch.log(degI);ldegO = torch.log(degO)
    #L = torch.sum(degI)
    #LperLink = L/(N**2 - N)
    # unifstartval =  0.5*torch.log((LperLink/((1-LperLink))))
    nnzInds = degIO != 0

#    if (torch.sum(degI) - torch.sum(degO) ) > 0.1 :
#        raise Exception("sums of in and out degrees should be equal")
    
    maxIt = 100
    it = 0
    vParUn = torch.cat((ldegI,ldegO))
    relErrMax = 10
    err_list = []
    while (relErrMax > targetErr) & (it<maxIt):
        vParUn = phiFunc(vParUn,ldegI,ldegO)
        relErrMax = torch.max(check_fit(vParUn,degIO,nnzInds))
        relErrMean = torch.mean(check_fit(vParUn,degIO,nnzInds))
        err_list.append((relErrMax.item(),relErrMean.item()) )
        it+=1
        #print(it)
        
    zeroParI,zeroParO = zeroDegParFun(degIO)
    
    vParOut =  vParUn.clone()
    vParOut[0:N][degIO[0:N]==0] = zeroParI
    vParOut[N:2*N][degIO[N:2*N]==0] = zeroParO

    if relErrMax > targetErr:
        print(relErrMax,torch.mean(check_fit(vParUn,degIO,nnzInds)))
        vParOut = estimate_NetModelDirBin1_MLE(degIO, vParOut, Steps=50, opt_n=1, plot_flag=plot_flag)

    return vParOut,it,err_list



#%% Test estimation of the 
N = 50
density = 0.5

Y_mat = gen_test_net(N,density)
A_mat = (Y_mat > 0).float()

degIO = strIO_from_mat(A_mat)
#test = jit(estimate_NetModelDirBin1)
vParOpt,it,err_list= estimate_NetModelDirBin1(degIO,plot_flag=True)


np.max(np.abs(strIO_from_mat(A_mat) - strIO_from_mat( expMat_NetModelDirBin1(vParOpt)) ))



 