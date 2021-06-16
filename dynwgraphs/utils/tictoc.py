#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 16th 2021

"""

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
