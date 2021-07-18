#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 23rd 2021

"""


"""

Functions for Zero Augmented models for sparse weighted dynamical networks, coded using pytorch
Author: Domenico Di Gangi


"""
import torch
import numpy as np
import pickle
import copy
from matplotlib import pyplot as plt
from .utils.tensortools import splitVec, tens, putZeroDiag, putZeroDiag_T, soft_lu_bound, strIO_from_mat
from .utils.opt import optim_torch
from pathlib import Path
from torch.autograd import grad
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

#--------------------------
# --------- Zero Augmented Static model functions
#self = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )

class dirGraphs_funs(nn.Module):
    """
    This class collects core methods used to model one observation of directed weighted sparse static networks, modelled with a zero augmented distribution, one  parameter per each node (hence the 1 in the name)
    """

    def __init__(self, avoid_ovflw_fun_flag=True, distr="",  size_dist_par_un_t=None, size_beta_t=None, like_type=None, phi_id_type=None):
        super().__init__()
        self.avoid_ovflw_fun_flag = avoid_ovflw_fun_flag
        self.distr = distr
        self.size_dist_par_un_t = size_dist_par_un_t
        self.size_beta_t = size_beta_t
        self.like_type = like_type
        self.ovflw_exp_L_limit = -50
        self.ovflw_exp_U_limit = 40
        self.phi_id_type=phi_id_type
        self.obs_type = "continuous_positive"


    def check_reg_and_coeff_pres(self, X, beta):
        """
        Check that X and beta are either both None or both not None. 
        """
        if (X is not None) and (beta is not None):
            return True
        elif (X is None) and (beta is None):
            return False 
        elif (X is not None) and (beta is None):
            raise 
        elif (X is None) and (beta is not None):
            raise 

    def check_reg_and_coeff_shape(self, X_t, beta_t):
        """
        Check that the shape of regressors and coefficients are correct and agree at every time step
        """
        if not (X_t.dim() == 3):
            raise
        if not (X_t.shape[0] == X_t.shape[1] == self.N):
            raise
        if beta_t.dim() !=2:
            raise

        if X_t.shape[2] != beta_t.shape[1]:
            raise Exception(f"X_t.shape = {X_t.shape} , beta_t.shape = {beta_t.shape} ")

        if beta_t.shape[0] not in [1, self.N, 2*self.N]:
            raise Exception(f"beta_t.shape = {beta_t.shape}, the allowed numers of parameters  beta_t for each regressors are [1, N, 2*N]" )

    def regr_product(self, beta_t, X_t):
        """
        Given a matrix of regressors and a vector of parameters obtain a matrix that has to be summed to the matrix obtained from nodes specific unrestricted dynamical paramters
        """
        if beta_t.shape[0] == 1:  # one parameter for all links
            prod = beta_t * X_t
        elif beta_t.shape[0] == self.N:
            # one parameter for each node for each regressor, i.e. same effect on in and out connections
            prod = beta_t * beta_t.unsqueeze(1) * X_t
        elif beta_t.shape[0] == 2*self.N:
            # two parameters (in and out connections) for each node for each regressor
            prod = (beta_t[:self.N, :] * beta_t[self.N:, :].unsqueeze(1)) * X_t

        return prod.sum(dim=2)

    def cond_exp_Y(self, phi, beta, X_t, ret_log=False, ret_all=False):
        """
        given the parameters related with the weights, compute the conditional expectation of matrix Y.
        the conditioning is on each matrix element being greater than zero
        can have non zero elements on the diagonal!!
        """
        phi_i, phi_o = splitVec(phi)
        log_Econd_mat = phi_i + phi_o.unsqueeze(1)
        if self.check_reg_and_coeff_pres(X_t, beta):
            log_Econd_mat = log_Econd_mat + self.regr_product(beta, X_t)

        if self.avoid_ovflw_fun_flag:
            """if required force the exponent to stay within overflow-safe bounds"""
            log_Econd_mat_restr = \
                soft_lu_bound(log_Econd_mat, l_limit=self.ovflw_exp_L_limit, u_limit=self.ovflw_exp_U_limit)
        else:
            log_Econd_mat_restr = log_Econd_mat

        if ret_all:
            return log_Econd_mat, log_Econd_mat_restr, torch.exp(log_Econd_mat_restr)
        elif ret_log:
            return log_Econd_mat_restr
            #  putZeroDiag(log_Econd_mat)
        else:
            return torch.exp(log_Econd_mat_restr)  
            #  putZeroDiag(torch.exp(log_Econd_mat))

    def identify_phi_io(self, phi):
        """ enforce an identification condition on the phi parameters for a single snapshot
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if self.phi_id_type == "first_in_zero":
            """set one to zero"""
            d = phi_i[0]
        elif self.phi_id_type == "in_sum_eq_out_sum":
            """set sum of in parameters equal to sum of out par """
            d = (phi_i.mean() - phi_o.mean())/2
        elif self.phi_id_type == "in_sum_zero":
            """set sum of in parameters to zero """
            d = phi_i.mean()
       
        phi_i_out = phi_i - d
        phi_o_out = phi_o + d
        return torch.cat((phi_i_out, phi_o_out))

    def identify_phi_io_beta(self, phi, beta, x):
        """ enforce an identification condition on the phi and beta parameters for a single snapshot, needed in case of uniform beta
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if self.phi_id_type != "in_sum_zero":
            raise
        else:
            if (~ torch.isclose(phi_i.mean(), torch.zeros(1), atol=1e-4)).any() :
                raise
        
        # raise error for cases where the identification problem is not yet clear
        if beta.shape[0] >1 : 
            raise
        if beta.shape[1] >1:
            raise

        d = phi_o.mean()
        phi_i_out = phi_i
        if x !=0 :
            phi_o_out = phi_o - d
            beta_out = beta + d/x 
        else:
            phi_o_out = phi_o
            beta_out = beta 

        return torch.cat((phi_i_out, phi_o_out)), beta_out

    def start_phi_from_obs(self, Y_t):
        pass

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        pass

    def loglike_t(self, Y_t, phi, beta, X_t, dist_par_un):
        """ The log likelihood of the zero augmented gamma network model as a function of the observations
        """
        pass

    def link_dist_par(self, dist_par_un, N, A_t=None):
        """
        take as input the unrestricted version of distribution parameters (the ones that we optimize)
        return their restricted versions, compatible with the different distributions
        """
        pass

    def dist_par_un_start_val(self):
        pass    
  
    def check_exp_vals(self, t):
        pass

    def str_from_dic(self, dict):
        return ''.join([f'{key}={value}_' for key, value in dict.items()])

    def file_names(self, save_path):
        pre_name =  save_path / f"{self.info_str_long()}"
        names = {}
        names["model"] = f"{pre_name}_model.pkl"
        names["parameters"] = f"{pre_name}_par.pkl"
        return names

    def save_model(self, save_path):
        file_name = self.file_names(save_path)["model"]
        pickle.dump(self, open(file_name, "wb"))

    def get_local_save_path(self):
        
        save_path = Path(f"./data_dynwgraphs_model")
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path


    def save_parameters(self, save_path=None):
        if save_path is None:
            save_path = self.get_local_save_path()
        file_name = self.file_names(save_path)["parameters"]
        pickle.dump(self.par_dict_to_save , open(file_name, "wb"))

    def par_matrix_T_to_list(self, mat):
        par_list = []
        T = mat.shape[1]
        for t in range(T):
            par_list.append(mat[:, t])
        return par_list

    def par_list_to_matrix_T(self, ls):
        return torch.stack([p.data for p in ls], dim=1)

    def get_t_or_0(self, t,  is_tv, par_list_T):
        if par_list_T is None:
            return None
        else:
            if is_tv:
                return par_list_T[t]
            elif len(par_list_T) == 1:
                return par_list_T[0]
            else:
                raise


    

    def sample_mats_from_par_lists(self, T, phi_T, beta_T=None, X_T=None, dist_par_un_T=None):

        if beta_T is not None:
            raise Exception("not yet ready")

        N = phi_T[0].shape[0]//2
        Y_T_sampled = torch.zeros(N, N, T)
        phi_tv = len(phi_T) >0
        if dist_par_un_T is not None:
            dist_par_tv = len(dist_par_un_T) >0
        else:
            dist_par_tv = None
        for t in range(T):
            phi_t = self.get_t_or_0(t, phi_tv, phi_T) 
            dist_par_un_t = self.get_t_or_0(t, dist_par_tv, dist_par_un_T) 
            
            dist = self.dist_from_pars(self.distr, phi_t , None, None, dist_par_un=dist_par_un_t, A_t=None)

            Y_T_sampled[:, :, t]  = dist.sample()
            
        return Y_T_sampled


class dirGraphs_sequence_ss(dirGraphs_funs):
    """
    Single snapshot sequence of, binary or weighted, fitness models with external regressors. 
    """

    _opt_options_ss_t_def = {"opt_n" :"ADAM", "min_opt_iter" :50, "max_opt_iter" :1000, "lr" :0.01}
    _opt_options_ss_seq_def = {"opt_n" :"ADAM", "min_opt_iter" :50, "max_opt_iter" :5000, "lr" :0.01}

    # set default init kwargs to be shared between binary and weighted models
    def __init__(self, Y_T, T_train=None, X_T=None, phi_tv=True, phi_par_init="fast_mle", avoid_ovflw_fun_flag=True, distr='',  phi_id_type="in_sum_eq_out_sum", like_type=None,  size_dist_par_un_t = None, dist_par_tv= None, size_beta_t = 1, beta_tv= tens([False]).bool(), data_name="", 
            opt_options_ss_t = _opt_options_ss_t_def, 
            opt_options_ss_seq = _opt_options_ss_seq_def):

        super().__init__(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distr=distr,  size_dist_par_un_t=size_dist_par_un_t, size_beta_t=size_beta_t, phi_id_type=phi_id_type, like_type=like_type)
        
        self.avoid_ovflw_fun_flag = avoid_ovflw_fun_flag
        
        self.Y_T = Y_T
        if T_train is not None:
            self.Y_T_train = Y_T[:,:,:T_train]
        else:
            self.Y_T_train = Y_T
            
        self.N = self.Y_T_train.shape[0]
        self.T_train = self.Y_T_train.shape[2]

        self.X_T = X_T
        if (T_train is not None) and (X_T is not None):
            self.X_T_train = X_T[:,:,:T_train]
        else:
            self.X_T_train = X_T
        
        if self.X_T_train is None:
            self.n_reg = 0
        else:
            self.n_reg = self.X_T_train.shape[2]

        self.phi_par_init = phi_par_init
        self.dist_par_tv = dist_par_tv
        self.beta_tv = beta_tv
        self.n_beta_tv = sum(beta_tv)
        self.phi_tv = phi_tv

        self.reg_cross_unique = self.check_regressors_cross_uniqueness()
        self.identification_type = ""
        self.check_id_required()
        
        self.opt_options_ss_t = opt_options_ss_t
        self.opt_options_ss_seq = opt_options_ss_seq
        
        self.data_name = data_name
        
        if not self.phi_tv:
            self.phi_T = [torch.zeros(self.N*2, requires_grad=True)]
            raise
        else:
            self.phi_T = [torch.zeros(self.N*2, requires_grad=True) for t in range(self.T_train)]

        if self.size_dist_par_un_t is None:
            self.dist_par_un_T = None
        else:
            if self.dist_par_tv:
                self.dist_par_un_T = [torch.zeros(size_dist_par_un_t, requires_grad=True) for t in range(self.T_train)]
            else:
                self.dist_par_un_T = [torch.zeros(size_dist_par_un_t, requires_grad=True)]
                
        if self.X_T_train is None:
            self.beta_T = None
            if any(beta_tv):
                raise
        else:
            if any(self.beta_tv):
                self.beta_T = [torch.zeros(size_beta_t, self.n_reg, requires_grad=True) for t in range(self.T_train)]
            else:
                self.beta_T = [torch.zeros(size_beta_t, self.n_reg, requires_grad=True)]

        self.optimized_once = False
        
    def get_obs_t(self, t):
        Y_t = self.Y_T_train[:, :, t]
        if self.X_T_train is not None:
            X_t = self.X_T_train[:, :, :, t]
        else:
            X_t = None
        return Y_t, X_t
 
    def get_par_t(self, t):
        """
        If a paramter is time varying, return the parameter at the t-th time step, if not return the only time step present. 
        beta_T can be None, in that case return beta_t = None
        """
        if not (0 <= t <= self.T_train) : 
            raise Exception(f"Requested t = {t}, T = {self.T_train} ")

        phi_t = self.get_t_or_0(t, self.phi_tv, self.phi_T)

        dist_par_un_t = self.get_t_or_0(t, self.dist_par_tv, self.dist_par_un_T)
    
        beta_t = self.get_t_or_0(t, any(self.beta_tv), self.beta_T)
       
        return phi_t, dist_par_un_t, beta_t

    def get_seq_latent_par(self):
        phi_T_data = self.par_list_to_matrix_T(self.phi_T) 
        
        if self.beta_T is not None:
            beta_T_data = self.par_list_to_matrix_T(self.beta_T)
            
        else:
            beta_T_data = None

        if self.dist_par_un_T is not None:
            dist_par_un_T_data = self.par_list_to_matrix_T(self.dist_par_un_T)
        else:
            dist_par_un_T_data = None

        return phi_T_data, dist_par_un_T_data, beta_T_data

    def check_id_required(self):

        if any(self.reg_cross_unique):
            if sum(self.reg_cross_unique) >1 :
                raise
            if any(self.beta_tv):
                if sum(self.beta_tv) >1 :
                    raise
                if self.reg_cross_unique[self.beta_tv]:
                    # tv beta for uniform regressor
                    self.identification_type = "phi_t_beta_t"#
                else:
                    raise # to be checked
                    self.identification_type = "phi_t"#

            else:
                #constant beta for uniform regressor
                self.identification_type = "phi_t_beta_const"#
        else:
            self.identification_type = "phi_t"#        

    def shift_sequence_phi_o_T_beta_const(self, c, x_T):    
        beta_t_all = self.beta_T[0].clone()
        beta_t_all[:, self.reg_cross_unique] = beta_t_all[:,self.reg_cross_unique] + c
        self.beta_T[0] = beta_t_all
        
        for t in range(self.T_train):
            phi_t = self.phi_T[t][:]
            phi_i, phi_o = splitVec(phi_t)
            phi_o = phi_o - c * x_T[t]
            self.phi_T[t] = torch.cat((phi_i, phi_o))

    def get_n_obs(self):
        return self.Y_T_train.numel()

    def get_n_par(self):
        n_par = 0

        if self.phi_tv:
            n_par += self.phi_T[0].numel() * len(self.phi_T)
        else:
            n_par += self.phi_T[0].numel()

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:
                n_par += self.phi_T[0].numel() * len(self.dist_par_un_T)
            else:
                n_par += self.phi_T[0].numel() * len(self.dist_par_un_T) * len(self.dist_par_un_T)

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:
                n_par += self.dist_par_un_T[0].numel() * len(self.dist_par_un_T)
            else:
                n_par += self.dist_par_un_T[0].numel() * len(self.dist_par_un_T) * len(self.dist_par_un_T)


        if self.beta_T is not None:
            if any(self.beta_tv):                    
                n_par += self.beta_T[0].numel() * len(self.beta_T)
                if any([not tv for tv in self.beta_tv]):
                    raise Exception("not ready to handle mixed beta behaviour, tv not tv")

            else:
                n_par += self.beta_T[0].numel() 

        return n_par

    def get_BIC(self):
        k = self.get_n_par()
        n = self.get_n_obs()
        logL = self.loglike_seq_T().detach()
        return  k * np.log(n) - 2 * logL

    def identify_sequence(self):    
        for t in range(self.T_train):
            #as first step, identify phi_io
            phi_t = self.phi_T[t][:]
            phi_t_id_0 = self.identify_phi_io(phi_t)
            
            if self.identification_type == "phi_t_beta_t":
                beta_t = self.beta_T[t][:, self.reg_cross_unique]
                x_t = self.X_T_train[0, 0, self.reg_cross_unique, t]
                self.phi_T[t][:], self.beta_T[t][:, self.reg_cross_unique] = self.identify_phi_io_beta(phi_t_id_0, beta_t, x_t )    
            else:    
                self.phi_T[t] = phi_t_id_0
    
        if self.identification_type == "phi_t_beta_const":
            phi_o_T_sum = sum([phi[self.N:].mean() for phi in self.phi_T ])
            if sum(self.reg_cross_unique) > 1:
                raise
            x_T = self.X_T_train[0, 0, self.reg_cross_unique, :].squeeze()
            x_T_sum = x_T.sum()
            c =(- self.beta_T[0][0]  + 1) # phi_o_T_sum/x_T_sum

            if len(self.beta_T)>1:
                raise

            self.shift_sequence_phi_o_T_beta_const(c, x_T)

    def check_regressors_seq_shape(self):
        """
        check that:
            - X_T_train is N x N x n_reg x T and 
            - beta_T is size_beta x n_reg x (T or 1) and 
        """
        if self.check_reg_and_coeff_pres(self.X_T_train, self.beta_T):
            for t in range(self.T_train-1):
                _, X_t = self.get_obs_t(t)
                _, _, beta_t = self.get_par_t(t)
                
                self.check_reg_and_coeff_shape(X_t, beta_t) 

    def check_regressors_cross_uniqueness(self):
        """
        return a bool tensor with one entry for each regressor. True if the regressor is uniform across links for more than a few time steps, false otherwise 
        """
        reg_cross_unique = torch.zeros(self.n_reg, dtype=bool)
        for k in range(self.n_reg):
            unique_flag_T = [(self.X_T_train[:,:,k,t].unique().shape[0] == 1) for t in range(self.T_train)]
            if all(unique_flag_T):
                reg_cross_unique[k] = True
            elif np.all(unique_flag_T) > 0.1:
                raise
        if reg_cross_unique.sum() > 1:
            raise
        return reg_cross_unique

    def run_checks(self):
       self.check_regressors_seq_shape()

    def set_all_requires_grad(self, list, flag):
        for p in list:
            p.requires_grad = flag

    def estimate_ss_t(self, t, est_phi, est_dist_par, est_beta, tb_log_flag=False):
        """
        single snapshot Maximum logLikelihood estimate of phi_t, dist_par_un_t and beta_t
        """

        Y_t, X_t = self.get_obs_t(t)
        
        par_l_to_opt = [] 

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        if est_phi:
            par_l_to_opt.append(phi_t)
        if est_dist_par:
            if dist_par_un_t is None:
                raise
            par_l_to_opt.append(dist_par_un_t)
        if est_beta:
            if beta_t is None:
                raise
            par_l_to_opt.append(beta_t)

        
        def obj_fun():
            phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
            return - self.loglike_t(Y_t, phi_t, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t)

        run_name = f"SS_t_{t}_"

        hparams_dict = {"time_step":t, "est_phi" :est_phi, "est_dist_par" :est_dist_par, "est_beta" :est_beta, "dist": self.distr, "size_par_dist": self.size_dist_par_un_t, "size_beta": self.size_beta_t, "avoid_ovflw_fun_flag":self.avoid_ovflw_fun_flag}

        optim_torch(obj_fun, list(par_l_to_opt), max_opt_iter=self.opt_options_ss_t["max_opt_iter"], opt_n=self.opt_options_ss_t["opt_n"], lr=self.opt_options_ss_t["lr"], min_opt_iter=self.opt_options_ss_t["min_opt_iter"], run_name=run_name, tb_log_flag=False, log_info=False, hparams_dict_in=hparams_dict)

    def loglike_seq_T(self):
        loglike_T = 0
        for t in range(self.T_train):
            Y_t, X_t = self.get_obs_t(t)
            
            phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
            
            loglike_T += self.loglike_t(Y_t, phi_t, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t)

        return loglike_T

    def init_phi_T_from_obs(self):
        if self.phi_tv:
            for t in range(self.T_train):
                if self.phi_par_init == "fast_mle":
                    Y_t, _ = self.get_obs_t(t)
                    self.phi_T[t] = self.start_phi_from_obs(Y_t)
                elif self.phi_par_init == "rand":
                    self.phi_T[t] = torch.rand(self.phi_T[t].shape)
                elif self.phi_par_init == "zeros":
                    self.phi_T[t] = torch.zeros(self.phi_T[t].shape)           
        else:
            raise

        self.identify_sequence()

    def plot_phi_T(self, x=None, i=None):
        if x is None:
            x = np.array(range(self.T_train))
        phi_T, _, _ = self.get_seq_latent_par()
        if i is not None:
            phi_i_T = (phi_T[:self.N,:])[i,:]
            phi_o_T = (phi_T[self.N:,:])[i,:]
        else:
            phi_i_T = phi_T[:self.N,:]
            phi_o_T = phi_T[self.N:,:]
        fig, ax = plt.subplots(2,1)
        ax[0].plot(x, phi_i_T.T)
        ax[1].plot(x, phi_o_T.T)
        return fig, ax
    
    def plot_beta_T(self, x=None):
        if x is None:
            x = np.array(range(self.T_train))
        _, _, beta_T = self.get_seq_latent_par()
        n_beta = beta_T.shape[2]
        fig, ax = plt.subplots(n_beta,1)
        if n_beta == 1:
            ax.plot(x, beta_T[:,:,0].T)
            ax.plot(x, beta_T[:,:,0].T)
        else:
            for i in range(n_beta):
                ax[i].plot(x, beta_T[:,:,i].T)
        return fig, ax
        
    def set_par_dict_to_save(self):
        self.par_dict_to_save = {}
        self.par_dict_to_save["phi_T"] = self.phi_T
        if self.dist_par_un_T is not None:
            self.par_dict_to_save["dist_par_un_T"] = self.dist_par_un_T
        if self.beta_T is not None:
            self.par_dict_to_save["beta_T"] = self.beta_T

        self.par_l_to_opt = []
        for l in self.par_dict_to_save.values():
            self.par_l_to_opt.extend(l)

    def estimate_ss_seq_joint(self, tb_log_flag=True, tb_save_fold="runs"):
        """
        Estimate from sequence of observations a set of parameters that can be time varying or constant. If time varying estimate a different set of parameters for each time step
        """
        
        self.run_checks()

        if not self.optimized_once:
            self.init_phi_T_from_obs()
        
        self.set_par_dict_to_save()

        self.optimized_once = True

        self.set_all_requires_grad(self.par_l_to_opt, True)

        def obj_fun():
            return  - self.loglike_seq_T()
      
        run_name = f"SSSeq_"

        opt_out = optim_torch(obj_fun, self.par_l_to_opt, max_opt_iter=self.opt_options_ss_seq["max_opt_iter"], opt_n=self.opt_options_ss_seq["opt_n"], lr=self.opt_options_ss_seq["lr"], min_opt_iter=self.opt_options_ss_seq["min_opt_iter"], run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = self.get_info_dict(), folder_name=tb_save_fold)

        self.identify_sequence()

        return opt_out

    def info_tv_par(self):
        return f"dist_{self.distr}_phi_tv_{self.phi_tv}_size_par_dist_{self.size_dist_par_un_t}_tv_{self.dist_par_tv}_size_beta_{self.size_beta_t}_tv_{self.beta_tv}_"

    def info_opt_par(self):
        return self.str_from_dic(self.opt_options_ss_seq)

    def info_filter(self):
        return  f"ss_"

    def info_str_long(self):
        return self.info_tv_par() + self.info_opt_par()

    def get_info_dict(self):

        model_info_dict = {"dist": self.distr, "filter_type":self.info_filter(), "phi_tv": float(self.phi_tv) if self.phi_tv is not None else 0, "size_par_dist": self.size_dist_par_un_t, "dist_par_tv": float(self.dist_par_tv) if self.dist_par_tv is not None else 0, "size_beta": self.size_beta_t, "beta_tv": self.beta_tv.sum().item(), "avoid_ovflw_fun_flag":self.avoid_ovflw_fun_flag}

        return model_info_dict




class dirGraphs_SD(dirGraphs_sequence_ss):
    """
        Version With Score Driven parameters.
        
        init_sd_type : "unc_mean", "est_joint", "est_ss_before"
    """

    __opt_options_sd_def = {"opt_n" :"ADAMHD", "min_opt_iter" :50, "max_opt_iter" :3500, "lr" :0.01}

    def __init__(self, *args, init_sd_type = "unc_mean", rescale_SD = True, opt_options_sd = __opt_options_sd_def, **kwargs ):

        super().__init__(*args, **kwargs)
        
        self.opt_options_sd = opt_options_sd
        self.rescale_SD = rescale_SD
        self.init_sd_type = init_sd_type
   
        self.max_value_A = 20
        self.B0 = self.re2un_B_par( torch.ones(1) * 0.98)
        self.A0 = self.re2un_A_par( torch.ones(1) * 0.0001)


        if self.phi_tv:
            self.sd_stat_par_un_phi = self.define_stat_un_sd_par_dict(self.phi_T[0].shape)    

        if self.dist_par_tv:
            self.sd_stat_par_un_dist_par_un = self.define_stat_un_sd_par_dict(self.dist_par_un_T[0].shape)    
           
        if self.beta_T is not None:    
            if any(self.beta_tv):
                self.sd_stat_par_un_beta = self.define_stat_un_sd_par_dict(self.beta_T[0].shape)    
            
                if not all(self.beta_tv):
                    raise


    def check_id_required(self):

        if any(self.reg_cross_unique):
            if sum(self.reg_cross_unique) >1 :
                raise
            if any(self.beta_tv):
                if sum(self.beta_tv) >1 :
                    raise
                
        self.identification_type = "phi_t"#
        
    def define_stat_un_sd_par_dict(self, n_sd_par):

        sd_stat_par_dict = {"w" :nn.Parameter(torch.zeros(n_sd_par)),"B" :nn.Parameter(torch.ones(n_sd_par)*self.B0),"A" :nn.Parameter(torch.ones(n_sd_par)*self.A0)}

        if self.init_sd_type != "unc_mean":
            sd_stat_par_dict["init_val"] = nn.Parameter(torch.zeros(n_sd_par))

        return nn.ParameterDict(sd_stat_par_dict)
    
    def un2re_B_par(self, B_un):
        exp_B = torch.exp(B_un)
        return torch.div(exp_B, (1 + exp_B))

    def un2re_A_par(self, A_un):
        exp_A = torch.exp(A_un)
        return torch.div(exp_A, (1 + exp_A)) * self.max_value_A

    def re2un_B_par(self, B_re):
        return torch.log(torch.div(B_re, 1 - B_re))

    def re2un_A_par(self, A_re):
        return torch.log(torch.div(A_re, self.max_value_A - A_re))

    def score_t(self, t):
        pass

    def update_dynw_par(self, t_to_be_updated):
        """
        score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
        """

        t = t_to_be_updated-1
      
        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        score_dict = self.score_t(t)

        if self.phi_tv:

            phi_t = self.identify_phi_io(phi_t)

            s = score_dict["phi"]
            w = self.sd_stat_par_un_phi["w"]
            B = self.un2re_B_par(self.sd_stat_par_un_phi["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_phi["A"])  
            phi_tp1 = w + torch.mul(B, phi_t) + torch.mul(A, s)

            # phi_tp1 = self.identify_phi_io(phi_tp1)

            self.phi_T[t+1] = phi_tp1

        if self.dist_par_tv:            

            s = score_dict["dist_par_un"]
            w = self.sd_stat_par_un_dist_par_un["w"]  
            B = self.un2re_B_par(self.sd_stat_par_un_dist_par_un["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_dist_par_un["A"])  

            dist_par_un_tp1 = w + torch.mul(B, dist_par_un_t) + torch.mul(A, s)

            self.dist_par_un_T[t+1] = dist_par_un_tp1

        if any(self.beta_tv):
    
            s = score_dict["beta"] 
            w = self.sd_stat_par_un_beta["w"]  
            B = self.un2re_B_par(self.sd_stat_par_un_beta["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_beta["A"])  
            
            beta_tp1 = w + torch.mul(B, beta_t) + torch.mul(A, s)

            # self.identify_phi_io_beta....
            self.beta_T[t+1] = beta_tp1

    def plot_sd_par(self):
        fig, ax = plt.subplots(3,1)
        ax[0].plot(self.sd_stat_par_un_phi["w"].detach())
        ax[1].plot(self.un2re_B_par(self.sd_stat_par_un_phi["B"].detach()))
        ax[2].plot(self.un2re_A_par(self.sd_stat_par_un_phi["A"].detach()))
        return fig, ax

    def get_unc_mean(self, sd_stat_par_un):
        w = sd_stat_par_un["w"]
        B = self.un2re_B_par(sd_stat_par_un["B"])  

        return w/(1-B)
        
    def set_unc_mean(self, unc_mean, sd_stat_par_un):
        B = self.un2re_B_par(sd_stat_par_un["B"])  
        w_implied_by_unc_mean = (1-B) * unc_mean 
        sd_stat_par_un["w"] = nn.parameter.Parameter(w_implied_by_unc_mean, requires_grad=True)
        
    def roll_sd_filt(self):

        """Use the static parameters and the observations, that are attributes of the class, to fiter, and update, the dynamical parameters with  score driven dynamics.
        """

        if self.init_sd_type == "unc_mean":
            if self.phi_tv:
                self.phi_T[0] = self.get_unc_mean(self.sd_stat_par_un_phi)
            if any(self.beta_tv):
                self.beta_T[0] = self.get_unc_mean(self.sd_stat_par_un_beta)
            if self.dist_par_tv:
                self.dist_par_un_T[0] = self.get_unc_mean(self.sd_stat_par_un_dist_par_un)
        elif self.init_sd_type in ["est_joint", "est_ss_before"]:
            if self.phi_tv:
                self.phi_T[0] = self.sd_stat_par_un_phi["init_val"]
            if any(self.beta_tv):
                self.beta_T[0] = self.sd_stat_par_un_beta["init_val"]
            if self.dist_par_tv:
                self.dist_par_un_T[0] = self.sd_stat_par_un_dist_par_un["init_val"] 
        else:
            raise

        for t in range(1, self.T_train):
            self.update_dynw_par(t)

        self.identify_sequence()

    def append_all_par_dict_to_list(self, par_dict, par_list, keys_to_exclude=[]):
        for k, v in par_dict.items():
            if k not in keys_to_exclude:
                par_list.append(v)

    def init_static_sd_from_obs(self):
       pass

    def info_filter(self):
        return f"sd_init_{self.init_sd_type}_resc_{self.rescale_SD}_"

    def info_opt_par(self):
        return self.str_from_dic(self.opt_options_sd)

    def set_par_val_from_dict(self, dict):
        for k, v in dict.items():
            logger.info(f"setting {k}")
            if not hasattr(self, k):
                raise Exception(f"attribute {k} not present")
            else:
                a = getattr(self, k)
                if type(a) != type(v):
                    if type(a) == list:
                        if type(a[0]) != type(v):
                            raise Exception(f"attribute {k} of wrong type")
                        else:
                            if not a[0].shape == v.shape:
                                raise Exception(f"attribute {k} of wrong shape")
                            else:
                                a[0] = v
                    else:
                        raise Exception(f"attribute {k} of wrong type")
                else:                    
                    if type(a) in [dict, torch.nn.modules.container.ParameterDict ]:
                        for k1, v1 in a.items():
                            logger.info(f"setting {k1}")
                            if not v1.shape == v[k1].shape:
                                raise Exception(f"attribute {k1} of wrong shape")
                        setattr(self, k, v)
                        
        self.par_dict_to_save = dict
        self.optimized_once = True
        self.roll_sd_filt()

    def set_par_dict_to_save(self):
        # dict define for consistency with non sd version
        self.par_dict_to_save = {}
        # define list of parameters to be optimized
        self.par_l_to_opt = [] 

        
        if self.phi_tv:
            self.par_dict_to_save["sd_stat_par_un_phi"] = self.sd_stat_par_un_phi
            if self.init_sd_type in ["est_ss_before", "unc_mean"]:
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_phi, self.par_l_to_opt, keys_to_exclude=par_to_exclude)
        else:
            self.par_l_to_opt.append(self.phi_T[0])
            self.par_dict_to_save["phi_T"] = self.phi_T[0]

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:
                self.par_dict_to_save["sd_stat_par_un_dist_par_un"] = self.sd_stat_par_un_dist_par_un

                if self.init_sd_type in ["est_ss_before", "unc_mean"]:
                    par_to_exclude = ["init_val"]
                else:
                    par_to_exclude = []
                self.append_all_par_dict_to_list(self.sd_stat_par_un_dist_par_un, self.par_l_to_opt, keys_to_exclude=par_to_exclude)

            else:
                self.par_l_to_opt.append(self.dist_par_un_T[0])
                self.par_dict_to_save["dist_par_un_T"] = self.dist_par_un_T[0]

        if any(self.beta_tv):
            self.par_dict_to_save["sd_stat_par_un_beta"] = self.sd_stat_par_un_beta
            
            if self.init_sd_type in ["est_ss_before", "unc_mean"]:
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_beta, self.par_l_to_opt, keys_to_exclude=par_to_exclude)
            
        elif self.beta_T is not None:
            self.par_l_to_opt.append(self.beta_T[0])
            self.par_dict_to_save["beta_T"] = self.beta_T[0]

    def estimate_sd(self, tb_log_flag=True, tb_save_fold="runs"):

        self.run_checks()

        if not self.optimized_once:
            self.init_static_sd_from_obs()
        self.optimized_once = True

        if self.init_sd_type == "est_ss_before":
            # the inititial value of sd tv par is etimated beforehand on a single snapshot

            self.estimate_ss_t(0, True, self.beta_T is not None, self.dist_par_un_T is not None)
       
        self.set_par_dict_to_save()

        def obj_fun():
            self.roll_sd_filt()
            return - self.loglike_seq_T()
          
        run_name = self.info_filter()

        return optim_torch(obj_fun, list(self.par_l_to_opt), max_opt_iter=self.opt_options_sd["max_opt_iter"], opt_n=self.opt_options_sd["opt_n"], lr=self.opt_options_sd["lr"], min_opt_iter=self.opt_options_sd["min_opt_iter"], run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = self.get_info_dict(), folder_name=tb_save_fold)

    def load_par(self,  load_path):
        par_dic = pickle.load(open(self.file_names(load_path)["parameters"], "rb"))

        self.set_par_val_from_dict(par_dic)

    def run(self, est_flag, l_s_path  ):
        if est_flag:
            logger.info("Estimating")
            self.estimate_sd()
            self.save_parameters(l_s_path)
        else:
            logger.info("Loading")
            self.load_par(l_s_path)

    def init_par_from_model_without_beta(self, mod_no_beta):
        self.sd_stat_par_un_phi = copy.deepcopy(mod_no_beta.sd_stat_par_un_phi)
        self.dist_par_un_T = copy.deepcopy(mod_no_beta.dist_par_un_T)
        self.optimized_once = True
        self.roll_sd_filt()
        assert mod_no_beta.loglike_seq_T() == self.loglike_seq_T()

    def init_par_from_model_with_const_par(self, lower_model):
        if lower_model.phi_tv:
            if self.phi_tv:
                if self.sd_stat_par_un_phi["w"].shape != lower_model.sd_stat_par_un_phi["w"].shape:
                    raise Exception("wrong shapes")

                self.sd_stat_par_un_phi = copy.deepcopy(lower_model.sd_stat_par_un_phi)
            else:
                raise Exception("lower model should have less dynamical parameters")
        else:
            if self.phi_tv:
                self.set_unc_mean(self.sd_stat_par_un_phi, lower_model.phi_T[0])
            else:
                self.phi_T = copy.deepcopy(lower_model.phi_T)


        if lower_model.dist_par_tv:
            if self.dist_par_tv:
                if self.sd_stat_par_un_dist_par_un["w"].shape != lower_model.sd_stat_par_un_dist_par_un["w"].shape:
                    raise Exception("wrong shapes")

                self.sd_stat_par_un_dist_par_un = copy.deepcopy(lower_model.sd_stat_par_un_dist_par_un)
            else:
                raise Exception("lower model should have less dynamical parameters")
        else:
            if self.dist_par_tv:
                self.set_unc_mean(self.sd_stat_par_un_dist_par_un, lower_model.dist_par_un_T[0])
            else:
                self.dist_par_un_T = copy.deepcopy(lower_model.dist_par_un_T)

        if any(lower_model.beta_tv):
            if any(self.beta_tv):
                if self.sd_stat_par_un_beta["w"].shape != lower_model.sd_stat_par_un_beta["w"].shape:
                    raise Exception("wrong shapes")

                self.sd_stat_par_un_beta = copy.deepcopy(lower_model.sd_stat_par_un_beta)
            else:
                raise Exception("lower model should have less dynamical parameters")
        else:
            if any(self.beta_tv):
                self.set_unc_mean(lower_model.beta_T[0], self.sd_stat_par_un_beta)
            else:
                self.beta_T = copy.deepcopy(lower_model.beta_T)
                            

        self.optimized_once = True
        
        self.roll_sd_filt()
        lower_model.roll_sd_filt()
        logger.info(f"lower model likelihood {lower_model.loglike_seq_T()}, model with sd par loglike {self.loglike_seq_T()}")

    def init_par_from_model_const_beta(self, mod_no_beta):
        self.sd_stat_par_un_phi = copy.deepcopy(mod_no_beta.sd_stat_par_un_phi)
        self.dist_par_un_T = copy.deepcopy(mod_no_beta.dist_par_un_T)
        self.optimized_once = True
        self.roll_sd_filt()
        assert mod_no_beta.loglike_seq_T() == self.loglike_seq_T()

    def get_n_par(self):
        n_par = 0
        if self.phi_tv:
            n_par += self.sd_stat_par_un_phi["w"].numel()
            n_par += self.sd_stat_par_un_phi["B"].numel()
            n_par += self.sd_stat_par_un_phi["A"].numel()
        else:
            n_par += self.phi_T[0].numel()

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:
                n_par += self.sd_stat_par_un_dist_par_un["w"].numel()
                n_par += self.sd_stat_par_un_dist_par_un["B"].numel()
                n_par += self.sd_stat_par_un_dist_par_un["A"].numel()
            else:
                n_par += self.dist_par_un_T[0].numel()


        if self.beta_T is not None:
            if any(self.beta_tv):                    
                n_par += self.sd_stat_par_un_beta["w"].numel()
                n_par += self.sd_stat_par_un_beta["B"].numel()
                n_par += self.sd_stat_par_un_beta["A"].numel()
                if any([not tv for tv in self.beta_tv]):
                    raise Exception("not ready to handle mixed beta behaviour, tv not tv")

            else:
                n_par += self.beta_T[0].numel()

        return n_par

    def get_info_dict(self):
        info_dict = super().get_info_dict()
        info_dict["init_sd_type"] = self.init_sd_type
        info_dict["rescale_SD"] = self.rescale_SD

        return info_dict

# Weighted Graphs

class dirSpW1_sequence_ss(dirGraphs_sequence_ss):
    
    def __init__(self, *args, distr="gamma", like_type=2,  size_dist_par_un_t = 1, dist_par_tv= False, **kwargs):

        super(dirSpW1_sequence_ss, self).__init__( *args, distr = distr, like_type=like_type,  size_dist_par_un_t = size_dist_par_un_t, dist_par_tv= dist_par_tv, **kwargs)

    def start_phi_from_obs(self, Y_t):
        N = Y_t.shape[0]
        A_t = tens(Y_t > 0)
        S_i = Y_t.sum(dim=0)
        S_o = Y_t.sum(dim=1)
        S_sqrt = S_i.sum().sqrt()
        K_i = A_t.sum(dim=0)
        K_o = A_t.sum(dim=1)
        phi_t_0_i = torch.log(S_i/S_sqrt )#* (N/K_i) )
        phi_t_0_o = torch.log(S_o/S_sqrt )#* (N/K_o) )

        phi_t_0 = torch.cat((phi_t_0_i, phi_t_0_o))

        max_val = phi_t_0[~torch.isnan(phi_t_0)].max()
        phi_t_0[torch.isnan(phi_t_0)] = -max_val
        phi_t_0[~torch.isfinite(phi_t_0)] = - max_val
        phi_t_0_i[K_i == 0] = - max_val
        phi_t_0_o[K_o == 0] = - max_val

        return self.identify_phi_io(phi_t_0)

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        N = phi.shape[0]//2
        # Restrict the distribution parameters.
        dist_par_re = self.link_dist_par(dist_par_un, N, A_t=A_t)
        if self.distr == 'gamma':
            EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
            if A_t is None:
                rate = torch.div(dist_par_re, EYcond_mat)
                distr_obj = torch.distributions.gamma.Gamma(dist_par_re, rate)
            else:# if A_t is given, we already took into account the dimension of dist_par above when restricting it
                rate = torch.div(dist_par_re, EYcond_mat[A_t])
                distr_obj = torch.distributions.gamma.Gamma(dist_par_re, rate)

        elif self.distr == 'lognormal':
            log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
            if A_t is None:
                sigma = dist_par_re
                mu = log_EYcond_mat - (sigma ** 2) / 2
                distr_obj = torch.distributions.log_normal.LogNormal(mu, sigma)
            else:  # if A_t is given, we already took into account the dimension of dist_par above when restricting it
                sigma = dist_par_re
                mu = log_EYcond_mat[A_t] - (sigma ** 2) / 2
                distr_obj = torch.distributions.log_normal.LogNormal(mu, sigma)
        return distr_obj

    def loglike_t(self, Y_t, phi, beta, X_t, dist_par_un):
        """ The log likelihood of the zero augmented gamma network model as a function of the observations
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t)
        A_t = Y_t > 0
        N = A_t.shape[0]
        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val()

        if (self.distr == 'gamma') and (self.like_type in [0, 1]):# if non torch computation of the likelihood is required
            # Restrict the distribution parameters.
            dist_par_re = self.link_dist_par(dist_par_un, N, A_t)
            if self.like_type==0:
                """ numerically stable version """
                log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(log_EYcond_mat[A_t])
                #tmp3 = - torch.sum(torch.div(Y_t[A_t], torch.exp(log_EYcond_mat[A_t])))
                tmp3 = - torch.sum(torch.exp(torch.log(Y_t[A_t])-log_EYcond_mat[A_t] + dist_par_re.log() ))
                out = tmp + tmp1 + tmp2 + tmp3
            elif self.like_type == 1:
                EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(torch.log(EYcond_mat[A_t]))
                tmp3 = - torch.sum(torch.div(Y_t[A_t], EYcond_mat[A_t])*dist_par_re)
                out = tmp + tmp1 + tmp2 + tmp3

        else:# compute the likelihood using torch buit in functions
            distr_obj = self.dist_from_pars(phi, beta, X_t, dist_par_un, A_t=A_t)
            log_probs = distr_obj.log_prob(Y_t[A_t])
            out = torch.sum(log_probs)        # softly bound loglikelihood from below??

        #out = soft_l_bound(out, -1e20)
        return out

    def link_dist_par(self, dist_par_un, N, A_t=None):
        """
        take as input the unrestricted version of distribution parameters (the ones that we optimize)
        return their restricted versions, compatible with the different distributions
        """
        if (self.distr == 'gamma') | (self.distr == 'lognormal'):
            if dist_par_un.shape[0] == 1:
                dist_par_re = torch.exp(dist_par_un)
            elif dist_par_un.shape[0] == N:
                dist_par_re = torch.exp(dist_par_un + dist_par_un.unsqueeze(1))
                if A_t is not None:
                    dist_par_re = dist_par_re[A_t]
        return dist_par_re

    def dist_par_un_start_val(self):
        if self.distr=='gamma':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(self.size_dist_par_un_t)
        elif self.distr=='lognormal':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(self.size_dist_par_un_t)
        return dist_par_un0
    
    def check_exp_vals(self, t,):
        
        Y_t, X_t = self.get_obs_t(t)
        phi_t, _, beta_t = self.get_par_t(t)
        EYcond_mat = self.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t)
        EYcond_mat = putZeroDiag(EYcond_mat)
        A_t=putZeroDiag(Y_t)>0
        return torch.sum(Y_t[A_t])/torch.sum(EYcond_mat[A_t]), torch.mean(Y_t[A_t]/EYcond_mat[A_t])

    def get_Y_T_to_save(self):
        return self.Y_T


class dirSpW1_SD(dirGraphs_SD, dirSpW1_sequence_ss):

    def __init__(self, *args, **kwargs):

        super(dirSpW1_SD, self).__init__(*args, **kwargs)

    def score_t(self, t):

        """
        given the observations and the ZA gamma parameters (i.e. the cond mean
        matrix and the dist_par_un par), return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """

        Y_t, X_t = self.get_obs_t(t)

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        A_t_bool = Y_t > 0
        A_t = tens(A_t_bool)
        
        score_dict = {}
        if  any(self.beta_tv) | self.dist_par_tv:
            
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t, dist_par_un=dist_par_un_t)

            if any(self.beta_tv):
                score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]
            if self.dist_par_tv:
                score_dict["dist_par_un"] = grad(like_t, dist_par_un_t, create_graph=True)[0]
            if self.rescale_SD:
                pass
                # raise "Rescaling not ready for beta and dist par"

        if self.phi_tv:
            sigma_mat = self.link_dist_par(dist_par_un_t, self.N)
            log_cond_exp_Y, log_cond_exp_Y_restr, cond_exp_Y = self.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t, ret_all=True)

            if self.distr == 'gamma':
                tmp = (Y_t.clone()/cond_exp_Y - A_t)*sigma_mat
                if self.rescale_SD:
                    diag_resc_mat = sigma_mat*A_t

            elif self.distr == 'lognormal':
                sigma2_mat = sigma_mat**2
                log_Y_t = torch.zeros(self.N, self.N)
                log_Y_t[A_t_bool] = Y_t[A_t_bool].log()
                tmp = (log_Y_t - log_cond_exp_Y_restr*A_t)/sigma2_mat + A_t/2
                if self.rescale_SD:
                    diag_resc_mat = (sigma_mat**2)**A_t

            if self.avoid_ovflw_fun_flag:
                L = self.ovflw_exp_L_limit
                U = self.ovflw_exp_U_limit
                # multiply the score by the derivative of the overflow limit function
                soft_bnd_der_mat = (1 - torch.tanh(2*((log_cond_exp_Y - L)/(L - U)) + 1)**2)
                tmp = soft_bnd_der_mat * tmp
                if self.rescale_SD:
                    diag_resc_mat = diag_resc_mat * (soft_bnd_der_mat**2)

            score_phi = strIO_from_mat(tmp)

            #rescale score if required
            if self.rescale_SD:
                diag_resc = strIO_from_mat(diag_resc_mat)
                diag_resc[diag_resc==0] = 1
                score_phi = score_phi/diag_resc.sqrt()

            score_dict["phi"] = score_phi

        return score_dict

    def init_static_sd_from_obs(self):
        T_init = 5
        for t in range(T_init):
            self.estimate_ss_t(t, True, True, False)
        phi_T, dist_par_un_T, beta_T = self.get_seq_latent_par()
        if self.phi_tv:
            phi_unc_mean_0 = phi_T[:, :T_init].mean(dim=1) 
            self.set_unc_mean(phi_unc_mean_0, self.sd_stat_par_un_phi)

# Binary Graphs

class dirBin1_sequence_ss(dirGraphs_sequence_ss):

    
    def __init__(self, Y_T_train, *args, **kwargs):

        Y_T_train = tens(Y_T_train > 0)

        super().__init__( Y_T_train, *args, distr = "bernoulli",   **kwargs)
  
        self.obs_type = "bin"


    def invPiMat(self, phi, beta, X_t, ret_log=False):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(phi_i + phi_j)
        """
        phi_i, phi_o = splitVec(phi)
        log_inv_pi_mat = phi_i + phi_o.unsqueeze(1)
        if X_t is not None:
            log_inv_pi_mat = log_inv_pi_mat + self.regr_product(beta, X_t)
        if self.avoid_ovflw_fun_flag:
            """if required force the exponent to stay whithin overflow-safe bounds"""
            log_inv_pi_mat =  soft_lu_bound(log_inv_pi_mat, l_limit=self.ovflw_exp_L_limit, u_limit=self.ovflw_exp_U_limit)

        if ret_log:
            return putZeroDiag(log_inv_pi_mat)
        else:
            return putZeroDiag(torch.exp(log_inv_pi_mat))

    def exp_A(self, phi, beta=None, X_t=None):
        """
        given the vector of unrestricted parameters, compute the expected
        matrix, can have non zero elements on the diagonal!!
        """
        invPiMat_= self.invPiMat(phi, beta=beta, X_t=X_t)
        out = invPiMat_/(1 + invPiMat_)
        return out

    def zero_deg_par_fun(self, Y_t, phi, method="AVGSPACING", degIO=None):
        """
        What should be the fitness of nodes with zero degree? The MLE is -inf which is not practical.
        """
        if method == "SMALL":
            #A small number equal for all nodes such that the probability of observing a link is very small
            zero_deg_par_i = -10
            zero_deg_par_o = -10

        elif method == "AVGSPACING":
            # the distance in fitnesses between degrees zero and one is equal to the average fitness distance corresponding
            # to one degree difference
            if degIO is None:
                degIO = strIO_from_mat(Y_t)
            if (sum(degIO) == 0) | (sum(phi) == 0):
                raise
            degI, degO = splitVec(degIO)
            par_i, par_o = splitVec(phi)
            imin = degI[degI != 0].argmin()
            omin = degO[degO != 0].argmin()
            # find a proxy of the one degree difference between fitnesses
            diffDegs = degI[degI != 0] - degI[degI != 0][imin]
            avgParStepI = torch.mean(((par_i[degI != 0] - par_i[degI != 0][imin])[diffDegs != 0]) /
                                     diffDegs[diffDegs != 0])

            if not torch.isfinite(avgParStepI):
                print(((par_i  - par_i[imin])[diffDegs != 0]) / diffDegs[diffDegs != 0])
                raise
            diffDegs = degO[degO  != 0] - degO[degO != 0][omin]
            avgParStepO = torch.mean((par_o[degO != 0] - par_o[degO != 0][omin])[diffDegs != 0] /
                                     diffDegs[diffDegs != 0])
            if not torch.isfinite(avgParStepO):
                print(((par_i - par_i[imin])[diffDegs != 0]) / diffDegs[diffDegs != 0])
                raise
            # subtract a multiple of this average step to fitnesses of smallest node
            zero_deg_par_i = par_i[degI != 0][imin] - avgParStepI * degI[degI != 0][imin]
            zero_deg_par_o = par_o[degO != 0][omin] - avgParStepO * degO[degO != 0][omin]

        elif method == "EXPECTATION":
            #  Set the fitness to a value s.t  the expectation
            # of the degrees for each node with deg zero is small
            N = phi.shape[0]//2
            max_prob = tens(0.01)
            if degIO is None:
                degIO = strIO_from_mat(Y_t)
            if (sum(degIO) == 0) | (sum(phi) == 0):
                raise
            par_i, par_o = splitVec(phi)
            zero_deg_par_i = - par_o.max() + torch.log(max_prob/N)
            zero_deg_par_o = - par_i.max() + torch.log(max_prob/N)

        return zero_deg_par_i, zero_deg_par_o

    def set_zero_deg_par(self, Y_t, phi_in, method="AVGSPACING", degIO=None):
        phi = phi_in.clone()
        if degIO is None:
            degIO = strIO_from_mat(Y_t)
        N = phi.shape[0]//2
        zero_deg_par_i, zero_deg_par_o = self.zero_deg_par_fun(Y_t, phi, method=method)
        phi[:N][degIO[:N] == 0] = zero_deg_par_i
        phi[N:][degIO[N:] == 0] = zero_deg_par_o
        return phi

    def phiFunc(self, phi, ldeg):
        ldeg_i, ldeg_o = splitVec(ldeg)
        phi_i, phi_o = splitVec(phi.exp())
        mat_i = putZeroDiag(1/(phi_i + (1/phi_o).unsqueeze(1)))
        mat_o = putZeroDiag(1/((1/phi_i) + phi_o.unsqueeze(1)))
        out_phi_i = (ldeg_i - torch.log(mat_i.sum(dim=0)))
        out_phi_o = (ldeg_o - torch.log(mat_o.sum(dim=1)))
        return torch.cat((out_phi_i, out_phi_o))

    def start_phi_from_obs(self, Y, n_iter=30, degIO=None):
        if degIO is None:
            degIO = tens(strIO_from_mat(Y))
        ldeg = degIO.log()
        nnzInds = degIO > 0
        phi_0 = torch.ones(degIO.shape[0]) * (-15)
        phi_0[nnzInds] = degIO[nnzInds].log()
        for i in range(n_iter):
            phi_0 = self.phiFunc(phi_0, ldeg)
            #phi_0[~nnzInds] = -15
        phi_0 = self.set_zero_deg_par(Y, phi_0, degIO=degIO)
        return phi_0.clone()

    def dist_from_pars(self, distr, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution matrix valued, from the model's parameters
        """
        p_mat = self.exp_A(phi, beta=beta, X_t=X_t)
        dist = torch.distributions.bernoulli.Bernoulli(p_mat)
        return dist

    def loglike_t(self, Y_t, phi, beta, X_t, degIO=None, **kwargs):
        """
        The log likelihood of the zero beta model with or without regressors
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t).clone()
        if (self.like_type == 0) and (beta is None):
            # if non torch computation of the likelihood is required (working only with no regressors)
            if degIO is None:
                degIO = strIO_from_mat(Y_t)
            tmp1 = torch.sum(phi * degIO)
            tmp2 = torch.sum(torch.log(1 + self.invPiMat(phi, beta=beta, X_t=X_t)))
            out = tmp1 - tmp2
        else:# compute the likelihood using torch buit in functions
            dist = self.dist_from_pars(self.distr, phi, beta, X_t, dist_par_un=None, A_t=None)
            log_probs = dist.log_prob(Y_t).clone()
            out = torch.sum(log_probs)
        return out.clone()

    def check_exp_vals(self, t):
        Y_t, X_t = self.get_obs_t(t)
        phi_t, _, beta_t = self.get_par_t(t)

        degIO = strIO_from_mat(Y_t)
        nnzInds = degIO != 0
        expMat = self.exp_A(phi_t, beta=beta_t, X_t=X_t)
        errsIO = (strIO_from_mat(expMat) - degIO)[nnzInds]
        relErrsIO = torch.div(torch.abs(errsIO), degIO[nnzInds])
        out = torch.abs(relErrsIO)  # check if the constraint is satisfied for all degs

        return out.mean()

    def get_Y_T_to_save(self):
        return self.Y_T >0
    
class dirBin1_SD(dirGraphs_SD, dirBin1_sequence_ss):

    def __init__(self, Y_T_train, **kwargs): 

        Y_T_train = tens(Y_T_train > 0)

        super().__init__( Y_T_train, **kwargs) 
  
    def score_t(self, t):
        Y_t, X_t = self.get_obs_t(t)

        phi_t, _, beta_t = self.get_par_t(t)

        A_t_bool = Y_t > 0
        A_t = tens(A_t_bool)
        
        score_dict = {}
        if  any(self.beta_tv) :
            
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t)

            if any(self.beta_tv):
                score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]
        
            if self.rescale_SD:
                pass
                # raise "Rescaling not ready for beta and dist par"

        if self.phi_tv:

            exp_A = self.exp_A(phi_t, beta=beta_t, X_t=X_t)

            tmp = A_t - exp_A

            if self.rescale_SD:
                diag_resc_mat = exp_A * (1 - exp_A)

            if self.avoid_ovflw_fun_flag:
                log_inv_pi__mat = self.invPiMat(phi_t, beta=beta_t, X_t=X_t, ret_log=True)
                L = self.ovflw_exp_L_limit
                U = self.ovflw_exp_U_limit
                # multiply the score by the derivative of the overflow limit function
                soft_bnd_der_mat = (1 - torch.tanh(2*((log_inv_pi__mat - L)/(L - U)) + 1)**2)
                tmp = soft_bnd_der_mat * tmp
                if self.rescale_SD:
                    diag_resc_mat = diag_resc_mat * (soft_bnd_der_mat**2)

            score_phi = strIO_from_mat(tmp)
     
            #rescale score if required
            if self.rescale_SD:
                diag_resc = torch.cat((diag_resc_mat.sum(dim=0), diag_resc_mat.sum(dim=1)))
                diag_resc[diag_resc==0] = 1
                score_phi = score_phi/diag_resc.sqrt()

            score_dict["phi"] = score_phi
    
        return score_dict
        
    def init_static_sd_from_obs(self):
        self.init_phi_T_from_obs()
        phi_T, dist_par_un_T, beta_T = self.get_seq_latent_par()
        if self.phi_tv:
            phi_unc_mean_0 = phi_T.mean(dim=1) 
            self.set_unc_mean(phi_unc_mean_0, self.sd_stat_par_un_phi)



