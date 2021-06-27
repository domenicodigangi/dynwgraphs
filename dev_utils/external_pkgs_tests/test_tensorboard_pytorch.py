#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday June 25th 2021

"""

#%%
#%%
from torch.utils.tensorboard import SummaryWriter
import numpy as np


w = SummaryWriter('logs') 
i = 1
    
for n_iter in range(100):
    w.add_scalar('Loss/train', np.random.random(), n_iter)

w.flush()
w.close()

# %%
