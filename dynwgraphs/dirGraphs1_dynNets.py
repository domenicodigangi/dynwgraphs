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
from .utils.tensortools import splitVec, tens, putZeroDiag, putZeroDiag_T, soft_lu_bound, strIO_from_mat, strIO_from_tens_T, size_from_str
from .utils.opt import optim_torch
from .utils.graph_properties import MatrixSymmetry
import itertools
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score

from pathlib import Path
from torch.autograd import grad
from torch.autograd.functional import hessian
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


# Classes with common methods for directed models, both weighted and binary

class dirGraphs_funs(nn.Module):
    """
    This class collects core methods used to model one observation of directed weighted sparse static networks, modelled with a zero augmented distribution, one  parameter per each node (hence the 1 in the name)
    """

    #TO DO: move some unused attributes to child class
    def __init__(self, avoid_ovflw_fun_flag=False, distr="", size_phi_t=None, size_dist_par_un_t=None, size_beta_t=None, like_type=None, par_vec_id_type=None):

        super().__init__()
        self.avoid_ovflw_fun_flag = avoid_ovflw_fun_flag
        self.distr = distr
        self.size_phi_t = size_phi_t
        self.size_dist_par_un_t = size_dist_par_un_t
        self.size_beta_t = size_beta_t
        self.like_type = like_type
        self.ovflw_exp_L_limit = -40
        self.ovflw_exp_U_limit = 30
        self.par_vec_id_type = par_vec_id_type
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
            prod = (beta_t * beta_t.unsqueeze(1)) * X_t
        elif beta_t.shape[0] == 2*self.N:
            # two parameters (in and out connections) for each node for each regressor
            prod = (beta_t[:self.N, :] + beta_t[self.N:, :].unsqueeze(1)) * X_t

        return prod.sum(dim=2)

    def get_phi_sum(self, phi):
        if phi is None:
            return 0
        else:
            if phi.shape[0] == 2*self.N:
                phi_i, phi_o = splitVec(phi)
                return phi_i + phi_o.unsqueeze(1)
            elif phi.shape[0] == self.N:
                raise "Not ready"
                return phi + phi.unsqueeze(1)
            else:
                raise


    def exp_of_fit_plus_reg(self, phi, beta, X_t, ret_log=False, ret_all=False):
        """
        given the vector of fitness, the regression coeff and the regressors, compute the exponential of the appropriate fitness and regressors sum
        """

        log_Econd_mat = self.get_phi_sum(phi)

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
        else:
            return torch.exp(log_Econd_mat_restr)  

    def exp_Y(self, phi, beta, X_t):
        pass

    def set_elements_from_par_tens_to(self, par_tens, inds, val):
        if par_tens.dim() == 1:
                return torch.where(inds, val, par_tens)
        elif par_tens.dim() == 2:
            if inds.dim() == 1:
                return torch.where(inds.unsqueeze(1).repeat_interleave(par_tens.shape[1], dim=1), val, par_tens)
            else:
                raise
        else:
            raise Exception("Do not know how to handle tensors of parameters with more than 2 dims")

    def identify_io_par_to_be_sum(self, par_vec, id_type, inds_to_excl=None):
        """ enforce an identification condition on input and output parameters that are going to enter the pdf always as par_in_i + par_out_j 
        """
        # set the first in parameter to zero
        if inds_to_excl is not None:
            if id_type == "first_in_zero":
                if inds_to_excl[0] == True:
                    raise Exception("cannot set first to zero and exclude it from identification as well")
                elif inds_to_excl[0] == False:
                    pass
                else:
                    raise Exception("If using non boolean indexing for inds_to_exclude, need to check that zero element is not being excluded")

            par_vec = self.set_elements_from_par_tens_to(par_vec, inds_to_excl, torch.tensor(0.0)) 

        par_vec_i, par_vec_o = splitVec(par_vec)

        if id_type == "first_in_zero":
            """set one to zero"""
            d = par_vec_i[0]
        elif id_type == "in_sum_eq_out_sum":
            """set sum of in parameters equal to sum of out par """
            d = (par_vec_i.mean() - par_vec_o.mean())/2
        elif id_type == "in_sum_zero":
            """set sum of in parameters to zero """
            d = par_vec_i.mean()

        par_vec_i_out = par_vec_i - d
        par_vec_o_out = par_vec_o + d

        par_vec_out = torch.cat((par_vec_i_out, par_vec_o_out))
        
        if inds_to_excl is not None:
            par_vec_out = self.set_elements_from_par_tens_to(par_vec, inds_to_excl, torch.tensor(0.0)) 

        return par_vec_out

    def identify_phi_io_and_const_unif_beta(self, phi, beta, x):
        """ enforce an identification condition on the phi and beta parameters for a single snapshot, needed in case of uniform beta
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if self.par_vec_id_type != "in_sum_zero":
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
        pre_name =  Path(save_path) / f"{self.info_str_long()}"
        names = {}
        names["model"] = f"{pre_name}model.pkl"
        names["parameters"] = f"{pre_name}par.pkl"
        return names

    def get_local_save_path(self):
        
        save_path = Path(f"./data_dynwgraphs_model")
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def save_parameters(self, save_path=None):
        if save_path is None:
            save_path = self.get_local_save_path()
        file_name = self.file_names(save_path)["parameters"]
        pickle.dump(self.par_dict_to_save , open(file_name, "wb"))

    def par_tens_T_to_list(self, ten):
        """
        convert a time series tensor into a list with one element per each time step. time is always assumed to be the last tens dim 
        """
        par_list = []
        if ten.dim() == 2:
            T = ten.shape[1]
            for t in range(T):
                par_list.append(ten[:, t])
        elif ten.dim() == 3:
            T = ten.shape[2]
            for t in range(T):
                par_list.append(ten[:, :, t])
            

        return par_list

    def par_list_to_tens_T(self, ls):
        """
        convert a time series list, with one element per each time step, into a tensor. time is always assumed to be the last tens dim 
        """

        if ls[0].dim() in [0, 1]:
            return torch.stack([p.data for p in ls], dim=1)
        elif ls[0].dim() == 2:
            return torch.stack([p.data for p in ls], dim=2)
        else:
            raise


    def list_to_time_ser(self, ls, T):
        if ls is not None:
            if len(ls) > 1:
                return self.par_list_to_tens_T(ls)[:, :T]
            else:
                ten = self.par_list_to_tens_T(ls)[:, :T]
                ten_T = ten.repeat_interleave(T, dim=ten.dim()-1)
                return ten_T
        else:
            return None

    def get_t_or_t0(self, t,  is_tv, par_list_T):
        if par_list_T is None:
            return None
        else:
            if is_tv:
                return par_list_T[t]
            elif len(par_list_T) == 1:
                return par_list_T[0]
            else:
                raise

    def get_reg_t_or_none(self, t, Z_T):
        """ regressors are assumed to always be in the form of 4d tensor N x N x N_regressors x T  """
        
        if Z_T is not None:
            Z_t = Z_T[:, :, :, t]
        else:
            Z_t = None
        return Z_t
    

    def sample_Y_T_from_par_list(self, T, phi_T, beta_T=None, X_T=None, dist_par_un_T=None, avoid_empty=True, A_T=None):

        if beta_T is not None:
            beta_tv = len(beta_T)>1
        else:
            beta_tv = None

        N = self.N
        Y_T_sampled = torch.zeros(N, N, T)
        if phi_T is not None:
            phi_tv = len(phi_T) >1
        else:
            phi_tv = None
        if dist_par_un_T is not None:
            dist_par_tv = len(dist_par_un_T) > 1
        else:
            dist_par_tv = None

        for t in range(T):
            X_t = self.get_reg_t_or_none(t, X_T)
            phi_t = self.get_t_or_t0(t, phi_tv, phi_T) 
            beta_t = self.get_t_or_t0(t, beta_tv, beta_T) 
            dist_par_un_t = self.get_t_or_t0(t, dist_par_tv, dist_par_un_T) 
            
            dist = self.dist_from_pars(phi_t , beta_t, X_t, dist_par_un=dist_par_un_t)

            Y_T_sampled[:, :, t]  = dist.sample()

        if avoid_empty:
            if A_T is not None:
                raise
            #  avoid snapshots having completely empty networks
            empty_T = Y_T_sampled.sum(dim=(0,1)) == 0
            if any(empty_T):
                logger.info(f"empty mat sampled for some t, adding links")
                
                deg_i_T , deg_o_T = splitVec(strIO_from_tens_T(Y_T_sampled))
                i_inds = deg_i_T.topk(3).indices
                o_inds = deg_o_T.topk(3).indices
                links = [(i, o) for i, o in itertools.product(i_inds, o_inds) if i!=o]
                logger.info(links)
                
                for t, is_empty in enumerate(empty_T):
                    if is_empty:
                        for io in links:
                            Y_T_sampled[io[0], io[1], t] = 1

        if A_T is not None:
            Y_T_sampled[~A_T] = 0
            
        return Y_T_sampled


class dirGraphs_sequence_ss(dirGraphs_funs):
    """
    Single snapshot sequence of, binary or weighted, fitness models with external regressors. 
    """

    _opt_options_ss_t_def = {"opt_n" :"ADAMHD", "max_opt_iter" :1000, "lr" :0.01, "disable_logging": True}
    _opt_options_ss_seq_def = {"opt_n" :"ADAMHD", "max_opt_iter" :15000, "lr" :0.01}

    # set default init kwargs to be shared between binary and weighted models
    def __init__(self, Y_T, T_train=None, X_T=None, phi_tv=True, phi_par_init_type="fast_mle", avoid_ovflw_fun_flag=True, distr='',  par_vec_id_type="in_sum_eq_out_sum", like_type=None, size_phi_t="2N",   size_dist_par_un_t = None, dist_par_tv= None, size_beta_t = None, beta_tv= tens([False]).bool(), beta_start_val=0, data_name="", 
            opt_options_ss_t = _opt_options_ss_t_def, opt_options_ss_seq = _opt_options_ss_seq_def, max_opt_iter = None, opt_n=None):

        self.avoid_ovflw_fun_flag = avoid_ovflw_fun_flag
        
        self.Y_T = Y_T
        self.T_all = self.Y_T.shape[2]
        # T_train is the index of the last obs in the training set
        if T_train is not None:
            self.T_train = int(T_train)
            if self.T_train == self.T_all:
                raise Exception("use T_train = None if using whole smaple")
        else:
            self.T_train = self.T_all

        # t_test_1 is the index of the first obs in the test (validation) set
        self.t_test_1 = self.T_train+1 
            
        self.N = self.Y_T.shape[0]

        super().__init__(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distr=distr, size_phi_t=self.size_from_str(size_phi_t), size_dist_par_un_t=self.size_from_str(size_dist_par_un_t), size_beta_t=self.size_from_str(size_beta_t), par_vec_id_type=par_vec_id_type, like_type=like_type)
        
        self.X_T = X_T

        if self.X_T is None:
            self.n_reg = 0
            self.beta_tv = None
            self.n_beta_tv = None
        else:
            self.n_reg = self.X_T.shape[2]
            self.beta_tv = self.tv_flag_list_from_input(beta_tv)

            self.n_beta_tv = sum(self.beta_tv)

        self.phi_par_init_type = phi_par_init_type
        self.phi_tv = self.tv_flag_bool_from_input(phi_tv)
        self.dist_par_tv = dist_par_tv
        self.beta_start_val = beta_start_val
        self.reg_cross_unique = self.check_regressors_cross_uniqueness()
        self.identif_multi_par = ""
        self.check_id_required()
        
        self.opt_options_ss_t = opt_options_ss_t
        self.opt_options_ss_seq = opt_options_ss_seq
        if max_opt_iter is not None:
            self.opt_options_ss_seq["max_opt_iter"] = max_opt_iter
        if opt_n is not None:
            self.opt_options_ss_seq["opt_n"] = opt_n
        self.opt_options = self.opt_options_ss_seq
        
        self.data_name = data_name
        
        self.model_symmetry = self.get_model_symmetry()
        self.init_all_par_sequences()

        self.check_types()
        self.inds_to_exclude_from_id = None


  
    def get_inds_inactive_nodes(self):
        return strIO_from_tens_T(self.Y_T) /self.Y_T.shape[2] < 0.05 

    def check_types(self):
        if self.beta_tv is not None:
            if type(self.beta_tv) in [list, torch.Tensor]:
                if type(self.beta_tv[0]) == bool:
                    pass
                elif type(self.beta_tv[0]) == torch.Tensor:
                    assert type(self.beta_tv[0].item()) == bool
                else:
                    raise TypeError()
            else:
                raise TypeError()
        
        if self.phi_tv is not None:
            if type(self.phi_tv) == bool:
                pass
            else:
                raise TypeError()
        
        if self.dist_par_tv is not None:
            if type(self.dist_par_tv) == bool:
                pass
            else:
                raise TypeError()
        
    def size_from_str(self, pox_str):
        return size_from_str(pox_str, self.N)

    def tv_flag_list_from_input(self, beta_tv):
        if type(beta_tv) == str:
            if ("[" in beta_tv) and ("]" in beta_tv):
                return eval(beta_tv)
            elif beta_tv in ["True", "False"]:
                return self.tv_flag_list_from_input(eval(beta_tv))
            else:
                raise Exception("String must encode a list")
        elif type(beta_tv) == bool:
            return tens([beta_tv for p in range(self.n_reg)] ).bool()
        elif type(beta_tv) in [list, torch.Tensor]:
            return tens(beta_tv).bool()
        else:
            raise Exception("Wrong type for beta_tv")
            
    def tv_flag_bool_from_input(self, tv_flag):
        if type(tv_flag) == str:
            return bool(eval(tv_flag))
        elif tv_flag in [True, False]:
            return tv_flag
        elif tv_flag == 1:
            return True
        elif tv_flag == 0:
            return False
        else:
            raise TypeError()

    def get_model_symmetry(self):
        self.check_size_phi()
        return MatrixSymmetry.dir

    def check_size_phi(self):
        if self.size_phi_t is None:
            pass
            # raise Exception("Model Without fitnesses not ready")
        elif self.size_phi_t == 0:
            pass
            # raise Exception("Model Without fitnesses not ready")
        elif self.size_phi_t == self.N:
            raise Exception("Undirected Model not ready")
        elif self.size_phi_t == 2*self.N:
            pass
        else:
            raise Exception(f"{self.size_phi_t} is not a valid choice")

    def init_all_par_sequences(self):   
        self.check_size_phi()

        if self.size_phi_t in [None, 0]:
            self.phi_T = None
        else:
            if not self.phi_tv:
                self.phi_T = [torch.zeros(self.size_phi_t, requires_grad=True)]
            else:
                self.phi_T = [torch.zeros(self.size_phi_t, requires_grad=True) for t in range(self.T_all)]

        if self.size_dist_par_un_t is None:
            self.dist_par_un_T = None
        else:
            if self.dist_par_tv:
                self.dist_par_un_T = [torch.zeros(self.size_dist_par_un_t, requires_grad=True) for t in range(self.T_all)]
            else:
                self.dist_par_un_T = [torch.zeros(self.size_dist_par_un_t, requires_grad=True)]
                
        if self.X_T is None:
            self.beta_T = None
            if (self.beta_tv is not None):
                if any(self.beta_tv):
                    raise
        else:
            if self.any_beta_tv():
                self.beta_T = [torch.ones(self.size_beta_t, self.n_reg, requires_grad=False) for t in range(self.T_all)]
            else:
                self.beta_T = [torch.ones(self.size_beta_t, self.n_reg, requires_grad=False)]

            for t, beta_t in enumerate(self.beta_T):
                self.beta_T[t] = beta_t * self.beta_start_val 

            for b in self.beta_T:
                b.requires_grad = True

        self.start_opt_from_current_par = False

    def set_elements_from_par_seq_to(self, par_seq, inds,  val):
        for t, par_tens in enumerate(par_seq):
            par_seq[t] = self.set_elements_from_par_tens_to(par_tens, inds, val) 

    def set_par_to_exclude_to_zero(self):
        if self.inds_to_exclude_from_id is not None:
            if self.size_phi_t == self.N:
                raise Exception("NEed to check how to handle nver observed fitnesses for undireceted networks")
            if self.size_phi_t == 2*self.N:
                self.set_elements_from_par_seq_to(self.phi_T, self.inds_to_exclude_from_id, torch.tensor(0.0))
            if self.size_beta_t == 2*self.N:
                self.set_elements_from_par_seq_to(self.beta_T, self.inds_to_exclude_from_id, torch.tensor(0.0))
            if self.size_dist_par_un_t == 2*self.N:
                self.set_elements_from_par_seq_to(self.beta_T, self.inds_to_exclude_from_id, torch.tensor(0.0))

    def get_Y_t(self, t):
        return self.Y_T[:,:,t]

    def get_X_t(self, t):
        return self.get_reg_t_or_none(t, self.X_T)
        
    def get_obs_t(self, t):
        Y_t = self.get_Y_t(t)
        X_t = self.get_X_t(t)
        return Y_t, X_t

    def get_train_obs_t(self, t):
        if t > self.T_train:
            raise
        else:
            return self.get_obs_t(t)

    def get_par_t(self, t):
        """
        If a paramter is time varying, return the parameter at the t-th time step, if not return the only time step present. 
        beta_T can be None, in that case return beta_t = None
        """
       

        phi_t = self.get_t_or_t0(t, self.phi_tv, self.phi_T)

        dist_par_un_t = self.get_t_or_t0(t, self.dist_par_tv, self.dist_par_un_T)
    
        beta_t = self.get_t_or_t0(t, self.any_beta_tv(), self.beta_T)
       
        return phi_t, dist_par_un_t, beta_t

    def get_train_Y_T(self):
        Y_T_train =  self.Y_T[:,:, :self.T_train]
        return Y_T_train
    
    def get_time_series_latent_par(self, only_train=False):
        if only_train:
            T = self.T_train
        else:
            T = self.T_all
        
        phi_T_data = self.list_to_time_ser(self.phi_T, T)
        dist_par_un_T_data = self.list_to_time_ser(self.dist_par_un_T, T)
        beta_T_data = self.list_to_time_ser(self.beta_T, T)

        return phi_T_data, dist_par_un_T_data, beta_T_data

    def any_beta_tv(self):
        if self.beta_tv is not None:
            # not ready to handle only some beta tv 
            assert any(self.beta_tv) == all(self.beta_tv)
            return any(self.beta_tv)
        else:
            return False

    def check_id_required(self):
        
        # do not identify beta for the moment
        self.identif_multi_par = "independent_id"#        

    def shift_sequence_phi_o_T_beta_const(self, c, x_T):    
        if self.reg_cross_unique.sum() > 1:
            raise
    
        beta_t_all = self.beta_T[0].clone()
        beta_t_all[:, self.reg_cross_unique] = beta_t_all[:,self.reg_cross_unique] + c
        self.beta_T[0] = beta_t_all
        
        for t in range(self.T_train):
            phi_t = self.phi_T[t][:]
            phi_i, phi_o = splitVec(phi_t)
            phi_o = phi_o - c * x_T[t]
            self.phi_T[t] = torch.cat((phi_i, phi_o))

    def get_n_obs(self):
        return self.Y_T[:, :, :self.T_train].numel()

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
            if self.any_beta_tv():                    
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

    def identify_par_vec(self, par_vec, id_type):
        if par_vec is None:
            return par_vec
        else:
            n_par = par_vec.shape[0]

            if n_par in [None, 1, self.N]:
                return par_vec
            elif n_par == 2* self.N:
                return self.identify_io_par_to_be_sum(par_vec, id_type, self.inds_to_exclude_from_id) 
            else:
                raise Exception(f"No identification setting for lenght = {n_par}")

    def identify_par_seq_T(self, par_seq):    
        if par_seq is not None:
            for t, par_vec in enumerate(par_seq):
                par_t = par_seq[t]
                par_t_id_0 = self.identify_par_vec(par_t, self.par_vec_id_type)
                par_seq[t] = par_t_id_0
                    
    def identify_sequences_phi_T_beta_const(self):    
        x_T = self.X_T[0, 0, self.reg_cross_unique, :].squeeze()
        phi_o_T_sum = sum([phi[self.N:].mean() for phi in self.phi_T ])
        x_T_sum = x_T.sum()
        for t in range(self.T_train):
            #as first step, identify phi_io
            phi_t = self.phi_T[t][:]
            phi_t_id_0 = self.identify_par_vec(phi_t, self.par_vec_id_type)     
            self.phi_T[t] = phi_t_id_0
    
            if sum(self.reg_cross_unique) > 1:
                raise
            c =(- self.beta_T[0][0]  + 1) # phi_o_T_sum/x_T_sum

            if len(self.beta_T)>1:
                raise

            self.shift_sequence_phi_o_T_beta_const(c, x_T)

    def identify_sequences(self, id_phi=True, id_beta=True, id_dist_par=True):
        if self.identif_multi_par == "independent_id":
            if id_phi:
                self.identify_par_seq_T(self.phi_T)
            if id_beta:
                self.identify_par_seq_T(self.beta_T)
            if id_dist_par:
                self.identify_par_seq_T(self.dist_par_un_T)

        if self.identif_multi_par == "phi_t_beta_const":
            raise Exception("need to check it before using it")

    def identify_sequences_tv_par(self):
        self.identify_sequences(id_phi=self.phi_tv, id_beta=self.any_beta_tv(), id_dist_par=self.dist_par_tv)


    def check_regressors_seq_shape(self):
        """
        check that:
            - X_T is N x N x n_reg x T and 
            - beta_T is size_beta x n_reg x (T or 1) and 
        """
        if self.check_reg_and_coeff_pres(self.X_T, self.beta_T):
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
            unique_flag_T = [(self.get_X_t(t)[:,:,k].unique().shape[0] == 1) for t in range(self.T_train)]
            if all(unique_flag_T):
                reg_cross_unique[k] = True
            elif np.all(unique_flag_T) > 0.1:
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

        Y_t, X_t = self.get_train_obs_t(t)
        
        par_l_to_opt = [] 

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)


        if est_phi:
            if phi_t is None:
                raise
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


        optim_torch(obj_fun, list(par_l_to_opt),**self.opt_options_ss_t, run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in=hparams_dict)

    def loglike_seq_T(self):
        loglike_T = 0
        if self.T_train < 2:
            raise Exception("likelihood from very short time series is unlikely to have much meaning")
        for t in range(self.T_train):
            Y_t, X_t = self.get_train_obs_t(t)
            
            phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
            
            loglike_T += self.loglike_t(Y_t, phi_t, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t)

        return loglike_T

    def init_phi_T_from_obs(self):
        with torch.no_grad():
            if self.phi_tv:
                for t in range(self.T_train):
                    if self.phi_par_init_type == "fast_mle":
                        Y_t, _ = self.get_train_obs_t(t)
                        self.phi_T[t] = self.start_phi_from_obs(Y_t)
                    elif self.phi_par_init_type == "rand":
                        self.phi_T[t] = torch.rand(self.phi_T[t].shape)
                    elif self.phi_par_init_type == "zeros":
                        self.phi_T[t] = torch.zeros(self.phi_T[t].shape)         

            else:
                Y_mean = self.Y_T.mean(dim=2)
                self.phi_T[0] = self.start_phi_from_obs(Y_mean)
                
            self.identify_sequences()

        for phi_t in self.phi_T:
            phi_t.requires_grad = True


    def plot_phi_T(self, x=None, i=None, fig_ax=None):
        if x is None:
            x = np.array(range(self.T_all))
        phi_T, _, _ = self.get_time_series_latent_par()
        if phi_T.shape[1] ==1:
            phi_T = phi_T.repeat_interleave(x.shape[0], dim=1)

        if self.inds_to_exclude_from_id is not None:
            phi_T[self.inds_to_exclude_from_id, :] = float('nan')

        if i is not None:
            phi_i_T = (phi_T[:self.N,:])[i,:]
            phi_o_T = (phi_T[self.N:,:])[i,:]
        else:

            phi_i_T = phi_T[:self.N,:]
            phi_o_T = phi_T[self.N:,:]
        if fig_ax is None:
            fig, ax = plt.subplots(2,1)
        else:
            fig, ax = fig_ax
        ax[0].plot(x, phi_i_T.T)
        ax[1].plot(x, phi_o_T.T)
        for a in ax:
            a.vlines(self.T_train, a.get_ylim()[0], a.get_ylim()[1], colors = "r", linestyles="dashed")
        return fig, ax
    
    def plot_beta_T(self, x=None, fig_ax = None):
        _, _, beta_T = self.get_time_series_latent_par()
        if x is None:
            x = np.array(range(self.T_all))
        n_beta = beta_T.shape[2]
        if fig_ax is None:
            fig, ax = plt.subplots(n_beta,1)
        else:
            fig, ax = fig_ax

        if n_beta == 1:
            if self.any_beta_tv():
                ax.plot(x, beta_T[:,:,0].T)
            else:
                ax.plot(x, beta_T[:,0,0])
        else:
            for i in range(n_beta):
                ax[i].plot(x, beta_T[:,:,i].T)
        return fig, ax
        
    def set_par_dict_to_opt_and_save(self):

        self.par_dict_to_save = {}
        if self.phi_T is not None:
            self.par_dict_to_save["phi_T"] = self.phi_T
            logger.info("adding phi_T to list of params to optimize")
        if self.dist_par_un_T is not None:
            logger.info("adding dist_par_un_T to list of params to optimize")
            self.par_dict_to_save["dist_par_un_T"] = self.dist_par_un_T
        if self.beta_T is not None:
            logger.info("adding beta_T to list of params to optimize")
            self.par_dict_to_save["beta_T"] = self.beta_T

        self.par_l_to_opt = []
        for l in self.par_dict_to_save.values():
            self.par_l_to_opt.extend(l)

    def estimate_ss_seq_joint(self, tb_log_flag=True, tb_save_fold="tb_logs"):
        """
        Estimate from sequence of observations a set of parameters that can be time varying or constant. If time varying estimate a different set of parameters for each time step
        """
        
        logger.info(self.opt_options)
        

        self.run_checks()

        if not self.start_opt_from_current_par:
            if self.phi_T is not None:
                self.init_phi_T_from_obs()

        self.set_par_dict_to_opt_and_save()

        self.start_opt_from_current_par = True

        def obj_fun():
            return  - self.loglike_seq_T()
      
        run_name = f"SSSeq_"

        opt_out = optim_torch(obj_fun, self.par_l_to_opt, **self.opt_options_ss_seq, run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = self.get_info_dict(), tb_folder=tb_save_fold)

        self.identify_sequences()

        return opt_out

    def estimate(self, **kwargs):
        return self.estimate_ss_seq_joint(**kwargs)

    def info_tv_par(self):
        return f"dist_{self.distr}_phi_tv_{self.phi_tv}_size_par_dist_{self.size_dist_par_un_t}_tv_{self.dist_par_tv}_size_beta_{self.size_beta_t}_tv_{[d.item() for d in self.beta_tv]}_"

    def info_opt_par(self):
        return self.str_from_dic(self.opt_options_ss_seq)

    def info_filter(self):
        return  f"ss_"

    def info_str_long(self):
        return self.info_filter() # self.info_tv_par() + self.info_opt_par()

    def get_info_dict(self):

        model_info_dict = {"dist": self.distr, "filter_type":self.info_filter(), "phi_tv": float(self.phi_tv) if self.phi_tv is not None else 0, "size_par_dist": self.size_dist_par_un_t, "dist_par_tv": float(self.dist_par_tv) if self.dist_par_tv is not None else 0, "size_beta": self.size_beta_t, "beta_tv": self.any_beta_tv(), "avoid_ovflw_fun_flag":self.avoid_ovflw_fun_flag}

        return model_info_dict

    def load_par(self,  load_path):
        logger.info("Loading par")
        if load_path[-4:] == ".pkl":
            file_name = load_path 
        else:
            file_name = self.file_names(load_path)["parameters"]
        
        par_dic = pickle.load(open(file_name, "rb"))

        self.set_par_val_from_dict(par_dic)

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
                            # logger.info(f"setting {k1}")
                            if not v1.shape == v[k1].shape:
                                raise Exception(f"attribute {k1} of wrong shape")
                        setattr(self, k, v)
                    elif type(a) == list:
                        for t in range(len(a)):
                            if a[t].shape == v[t].shape:
                                a[t] = v[t]
                            else: 
                                raise Exception(f"attribute {k} of wrong shape {a[t].shape}, {v[t].shape}")
                    else:
                        raise
                        
        self.par_dict_to_save = dict
        self.start_opt_from_current_par = True

    def out_of_sample_eval(self):
        logger.warn(f"Out of sample eval for ss sequences not ready. returning ")
        return {"auc_score":0}

    def set_inds_to_exclude_from_id(self):
        pass

    def sample_Y_T(self):
        pass

    def sample_and_set_Y_T(self, **kwargs):
        self.Y_T = self.sample_Y_T(**kwargs)
        self.set_inds_to_exclude_from_id()

    def get_avg_beta_dict(self):
        if self.beta_T is not None:
            _,  _, beta_T = self.get_time_series_latent_par()
            avg_beta_dict = {} 
            avg_beta_vec = beta_T.mean(dim=(0, 2)).detach().numpy()
            for n in range(avg_beta_vec.shape[0]):
                avg_beta_dict[f"avg_beta_{n+1}"] = avg_beta_vec[n]
        else:
            avg_beta_dict = {"avg_beta_1": float('nan')}
        return avg_beta_dict

    
    def score_t(self, t, rescale_flag, phi_flag, dist_par_un_flag, beta_flag):
        pass




class dirGraphs_SD(dirGraphs_sequence_ss):
    """
        Version With Score Driven parameters.
        
        init_sd_type : "unc_mean", "est_joint", "est_ss_before"
    """

    __opt_options_sd_def = {"opt_n" :"ADAMHD", "max_opt_iter" :15000, "lr" :0.01}

    __max_value_A = 20
    __max_value_B = 1 - 1e-3
    __B0 = torch.ones(1) * 0.98
    __A0 = torch.ones(1) * 0.00001
    __A0_beta = torch.ones(1) * 1e-12


    def __init__(self, Y_T, init_sd_type = "est_ss_before", rescale_SD = True, opt_options_sd = __opt_options_sd_def, **kwargs ):

        
        super().__init__(Y_T, **kwargs)
        self.opt_options_sd = opt_options_sd
        if "max_opt_iter" in kwargs.keys():
            self.opt_options_sd["max_opt_iter"] = kwargs["max_opt_iter"]
        if "opt_n" in kwargs.keys():
            self.opt_options_sd["opt_n"] = kwargs["opt_n"]
        self.opt_options = self.opt_options_sd

        self.rescale_SD = rescale_SD
        self.init_sd_type = init_sd_type

        self.init_all_stat_par()
   
    def remove_sd_specific_keys(self, dic):
        dic.pop("init_sd_type", None)
        dic.pop("rescale_SD", None)
        dic.pop("opt_options_sd", None)
        return dic        

    def init_all_stat_par(self, B0_un=None, A0_un=None, max_value_B=None, max_value_A=None):

        self.init_all_par_sequences()

        
        if max_value_B is None:
            self.max_value_B = self.__max_value_B

        if max_value_A is None:
            self.max_value_A = self.__max_value_A

        if B0_un is None:
            self.B0_un = self.re2un_B_par( torch.ones(1) * self.__B0)
        else:
            self.B0_un = self.re2un_B_par( torch.ones(1) * B0_un)
        if A0_un is None:
            self.A0_un = self.re2un_A_par( torch.ones(1) * self.__A0)
            self.A0_beta_un = self.re2un_A_par( torch.ones(1) * self.__A0_beta)
        else:
            self.A0_un = self.re2un_A_par( torch.ones(1) * A0_un)
            self.A0_beta_un = self.re2un_A_par( torch.ones(1) * A0_un)

        if self.phi_T is not None:
            if self.phi_tv:
                self.sd_stat_par_un_phi = self.define_stat_un_sd_par_dict(self.phi_T[0].shape, self.B0_un, self.A0_un)    

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:
                self.sd_stat_par_un_dist_par_un = self.define_stat_un_sd_par_dict(self.dist_par_un_T[0].shap, self.B0_un, self.A0_une)    
           
        if self.beta_T is not None:    
            if self.any_beta_tv():
                self.sd_stat_par_un_beta = self.define_stat_un_sd_par_dict(self.beta_T[0].shape, self.B0_un, self.A0_beta_un)    
        
        self.start_opt_from_current_par = False
                
    def define_stat_un_sd_par_dict(self, n_sd_par, B0_un, A0_un):

        sd_stat_par_dict = {"w" :nn.Parameter(torch.zeros(n_sd_par)),"B" :nn.Parameter(torch.ones(n_sd_par)*B0_un),"A" :nn.Parameter(torch.ones(n_sd_par)*A0_un)}

        if self.init_sd_type != "unc_mean":
            sd_stat_par_dict["init_val"] = nn.Parameter(torch.zeros(n_sd_par))

        return nn.ParameterDict(sd_stat_par_dict)
    
    def un2re_B_par(self, B_un):
        exp_B = torch.exp(B_un)
        return torch.div(exp_B, (1 + exp_B)) * self.max_value_B

    def un2re_A_par(self, A_un):
        exp_A = torch.exp(A_un)
        return torch.div(exp_A, (1 + exp_A)) * self.max_value_A

    def re2un_B_par(self, B_re):
        return torch.log(torch.div(B_re, self.max_value_B - B_re))

    def re2un_A_par(self, A_re):
        return torch.log(torch.div(A_re, self.max_value_A - A_re))


    def update_dynw_par(self, t_to_be_updated):
        """
        score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
        """

        t = t_to_be_updated-1
      
        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        score_dict = self.score_t(t, self.rescale_SD, self.phi_tv, self.dist_par_tv, self.any_beta_tv())

        if self.phi_T is not None:
            if self.phi_tv:    
                s = score_dict["phi"]
                w = self.sd_stat_par_un_phi["w"]
                B = self.un2re_B_par(self.sd_stat_par_un_phi["B"])  
                A = self.un2re_A_par(self.sd_stat_par_un_phi["A"])  
                phi_tp1 = w + torch.mul(B, phi_t) + torch.mul(A, s)

                self.phi_T[t+1] = phi_tp1

        if self.dist_par_un_T is not None:
            if self.dist_par_tv:            

                s = score_dict["dist_par_un"]
                w = self.sd_stat_par_un_dist_par_un["w"]  
                B = self.un2re_B_par(self.sd_stat_par_un_dist_par_un["B"])  
                A = self.un2re_A_par(self.sd_stat_par_un_dist_par_un["A"])  

                dist_par_un_tp1 = w + torch.mul(B, dist_par_un_t) + torch.mul(A, s)

                self.dist_par_un_T[t+1] = dist_par_un_tp1

        if self.beta_T is not None:
            if self.any_beta_tv():
        
                s = score_dict["beta"] 
                w = self.sd_stat_par_un_beta["w"]  
                B = self.un2re_B_par(self.sd_stat_par_un_beta["B"])  
                A = self.un2re_A_par(self.sd_stat_par_un_beta["A"])  
                
                beta_tp1 = w + torch.mul(B, beta_t) + torch.mul(A, s)

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
        
    def roll_sd_filt(self, T_last):

        """Use the static parameters and the observations, that are attributes of the class, to fiter, and update, the dynamical parameters with  score driven dynamics.
        """

        if self.init_sd_type == "unc_mean":
            if self.phi_T is not None:
                if self.phi_tv:
                    self.phi_T[0] = self.get_unc_mean(self.sd_stat_par_un_phi)

            if self.beta_T is not None:
                if self.any_beta_tv():
                    self.beta_T[0] = self.get_unc_mean(self.sd_stat_par_un_beta)

            if self.dist_par_un_T is not None:
                if self.dist_par_tv:
                    self.dist_par_un_T[0] = self.get_unc_mean(self.sd_stat_par_un_dist_par_un)

        elif self.init_sd_type in ["est_joint", "est_ss_before"]:
            if self.phi_T is not None:
                if self.phi_tv:
                    self.phi_T[0] = self.sd_stat_par_un_phi["init_val"]

            if self.beta_T is not None:
                if self.any_beta_tv():
                    self.beta_T[0] = self.sd_stat_par_un_beta["init_val"]

            if self.dist_par_un_T is not None:
                if self.dist_par_tv:
                    self.dist_par_un_T[0] = self.sd_stat_par_un_dist_par_un["init_val"] 
        else:
            raise

        for t in range(1, T_last):
            self.update_dynw_par(t)

        self.identify_sequences_tv_par()
                
    def roll_sd_filt_train(self):
        self.roll_sd_filt(self.T_train)

    def append_all_par_dict_to_list(self, par_dict, par_list, keys_to_exclude=[]):
        for k, v in par_dict.items():
            if k not in keys_to_exclude:
                par_list.append(v)
  

    def init_one_set_sd_par(self, sd_stat_par_un, par_0):
        if par_0 is not None:
            self.set_unc_mean(par_0, sd_stat_par_un)
            if self.init_sd_type == "est_joint":
                sd_stat_par_un["init_val"] = nn.parameter.Parameter(par_0)
            elif self.init_sd_type == "est_ss_before":
                sd_stat_par_un["init_val"] = nn.parameter.Parameter(par_0, requires_grad=False)


    def init_static_sd_from_obs(self):

        T_init = 5

        self.mod_stat.T_train = T_init

        logger.info("init static estimate to initialize sd parameters")

        self.mod_stat.estimate_ss_seq_joint(tb_log_flag=False)

        
        if self.init_sd_type not in ["unc_mean", "est_ss_before", "est_joint"]:
            raise Exception("unknown initialization")

        if self.phi_T is not None:
            if self.phi_tv:
                self.init_one_set_sd_par(self.sd_stat_par_un_phi, self.mod_stat.phi_T[0])
        if self.dist_par_un_T is not None:
            if self.dist_par_un_tv:
                self.init_one_set_sd_par(self.sd_stat_par_un_dist_par_un, self.mod_stat.dist_par_un_T[0])
        if self.beta_T is not None:
            if self.beta_tv:
                self.init_one_set_sd_par(self.sd_stat_par_un_beta, self.mod_stat.beta_T[0])
        
        self.roll_sd_filt_train()


    def info_filter(self):
        return f"sd_init_{self.init_sd_type}_resc_{self.rescale_SD}_"

    def info_opt_par(self):
        return self.str_from_dic(self.opt_options_sd)

    def set_par_dict_to_opt_and_save(self):
        # dict define for consistency with non sd version
        self.par_dict_to_save = {}
        # define list of parameters to be optimized
        self.par_l_to_opt = [] 
       
        
        if self.phi_T is not None:
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

        
        if self.any_beta_tv():
            self.par_dict_to_save["sd_stat_par_un_beta"] = self.sd_stat_par_un_beta
            
            if self.init_sd_type in ["est_ss_before", "unc_mean"]:
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_beta, self.par_l_to_opt, keys_to_exclude=par_to_exclude)
            
        elif self.beta_T is not None:
            self.par_l_to_opt.append(self.beta_T[0])
            self.par_dict_to_save["beta_T"] = self.beta_T[0]

    def estimate_sd(self, tb_log_flag=True, tb_save_fold="tb_logs"):
        logger.info(self.opt_options)
        self.run_checks()

        if not self.start_opt_from_current_par:
            self.init_static_sd_from_obs()
        self.start_opt_from_current_par = True

        self.set_par_dict_to_opt_and_save()

        def obj_fun():
            self.roll_sd_filt_train()
            return - self.loglike_seq_T()
          
        
        run_name = self.info_filter()
        
        opt_res = optim_torch(obj_fun, list(self.par_l_to_opt), **self.opt_options_sd, run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = self.get_info_dict(), tb_folder=tb_save_fold)

      
        return opt_res
        
    def estimate(self, **kwargs):
        return self.estimate_sd(**kwargs)
      
    def load_or_est(self, est_flag, l_s_path  ):
        if est_flag:
            logger.info("Estimating")
            self.estimate_sd()
            self.save_parameters(l_s_path)
        else:
            logger.info("Loading")
            self.load_par(l_s_path)
 
    def init_par_from_lower_model(self, lower_model):
        """
        For each set of par (phi, dist_par_un, beta), check if present in lower. if so use it to set corresponding self par
        """

        logger.info("init par from lower model")
 
        if lower_model.phi_T is not None:
            if lower_model.phi_tv:
                if self.phi_tv:
                    if self.sd_stat_par_un_phi["w"].shape != lower_model.sd_stat_par_un_phi["w"].shape:
                        raise Exception("wrong shapes")

                    self.sd_stat_par_un_phi = copy.deepcopy(lower_model.sd_stat_par_un_phi)
                else:
                    raise Exception("lower model should have less dynamical parameters")
            else:
                if self.phi_tv:
                    self.set_unc_mean(lower_model.phi_T[0], self.sd_stat_par_un_phi)
                else:
                    self.phi_T = copy.deepcopy(lower_model.phi_T)

        if lower_model.dist_par_un_T is not None:
            if lower_model.dist_par_tv:
                if self.dist_par_tv:
                    if self.sd_stat_par_un_dist_par_un["w"].shape != lower_model.sd_stat_par_un_dist_par_un["w"].shape:
                        raise Exception("wrong shapes")

                    self.sd_stat_par_un_dist_par_un = copy.deepcopy(lower_model.sd_stat_par_un_dist_par_un)
                else:
                    raise Exception("lower model should have less dynamical parameters")
            else:
                if self.dist_par_tv:
                    self.set_unc_mean(lower_model.dist_par_un_T[0], self.sd_stat_par_un_dist_par_un)
                else:
                    self.dist_par_un_T = copy.deepcopy(lower_model.dist_par_un_T)


        if lower_model.beta_T is not None:
            if lower_model.any_beta_tv():
                if self.any_beta_tv():
                    if self.sd_stat_par_un_beta["w"].shape != lower_model.sd_stat_par_un_beta["w"].shape:
                        raise Exception("wrong shapes")

                    self.sd_stat_par_un_beta = copy.deepcopy(lower_model.sd_stat_par_un_beta)
                else:
                    raise Exception("lower model should have less dynamical parameters")
            else:
                if self.any_beta_tv():
                    self.set_unc_mean(lower_model.beta_T[0], self.sd_stat_par_un_beta)
                else:
                    self.beta_T = copy.deepcopy(lower_model.beta_T)

        self.start_opt_from_current_par = True
        
        self.roll_sd_filt_train()
        lower_model.roll_sd_filt_train()
        try:
            assert torch.isclose(lower_model.loglike_seq_T(), self.loglike_seq_T())
            logger.info(f"lower model likelihood {lower_model.loglike_seq_T()}, upper model likelihood {self.loglike_seq_T()}")
        except:
            logger.error(f"lower model likelihood {lower_model.loglike_seq_T()}, upper model likelihood {self.loglike_seq_T()}")

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
            if self.any_beta_tv():                    
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

    def load_par(self, load_path):
        super().load_par(load_path)
        self.roll_sd_filt(T_last=self.T_all)

    def get_forecast(self, t):
        if t<= self.T_train:
            logger.error(f"Should forecast time steps not in the train set T_train {self.T_train}, required {t}")
            raise
     
        # Score driven parameters at time t are actually forecasts, since they are computed using obs up to t-1
        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
        X_t = self.get_X_t(t)
        
        F_A_t = self.exp_Y(phi_t, beta=beta_t, X_t=X_t)
        return F_A_t

    def get_out_of_sample_obs_and_pred(self, only_present, inds_keep_subset = None):

        if inds_keep_subset is None:
            inds_keep_subset = torch.ones(self.N, self.N, dtype=bool)
        else:
            if only_present:
                raise

        self.roll_sd_filt(self.T_all)

        F_Y_vec_all = np.zeros(0)
        Y_vec_all = np.zeros(0)

        for t in range(self.t_test_1, self.T_all): 
            
            Y_t = self.get_Y_t(t)
            Y_vec_t = Y_t.detach().numpy()
            F_Y_vec_t = self.get_forecast(t).detach().numpy()

            if only_present:
                inds_keep_subset = Y_t>0            
            Y_vec_all = np.append(Y_vec_all, Y_vec_t[inds_keep_subset])
            F_Y_vec_all = np.append(F_Y_vec_all, F_Y_vec_t[inds_keep_subset])
            
        return Y_vec_all, F_Y_vec_all

    def out_of_sample_eval(self):
        pass

# Weighted Graphs

class dirSpW1_sequence_ss(dirGraphs_sequence_ss):

    def __init__(self, Y_T, distr="gamma", like_type=2,  size_dist_par_un_t = 1, dist_par_tv= False, avoid_ovflw_fun_flag=True, **kwargs):
        
        super().__init__( Y_T, distr = distr, like_type=like_type,  size_dist_par_un_t = size_dist_par_un_t, dist_par_tv= dist_par_tv, **kwargs)

        self.model_class = "dirSpW1"
        self.bin_mod = dirBin1_sequence_ss(torch.zeros(10, 10, 20), size_phi_t = "2N" )

        self.set_inds_to_exclude_from_id()


    def set_inds_to_exclude_from_id(self):
        self.inds_to_exclude_from_id = strIO_from_tens_T(self.Y_T) == 0
        if all(self.inds_to_exclude_from_id):
            logger.warning("All parameters would be ignored. Assuming that this is a test model and setting inds_to_exclude_from_id = None")
            self.inds_to_exclude_from_id = None
        
        self.set_par_to_exclude_to_zero()

                        

    def exp_Y(self, phi, beta, X_t):
        return self.exp_of_fit_plus_reg(phi, beta=beta, X_t=X_t)

    def start_phi_from_obs(self, Y_t):
        N = Y_t.shape[0]
        A_t = tens(Y_t > 0)
        S_i = Y_t.sum(dim=0)
        S_o = Y_t.sum(dim=1)
        S_sqrt = S_i.sum().sqrt()
        K_i = A_t.sum(dim=0)
        K_o = A_t.sum(dim=1)
        if self.size_phi_t == 2*self.N:
            phi_t_0_i = torch.log(S_i/S_sqrt )#* (N/K_i) )
            phi_t_0_o = torch.log(S_o/S_sqrt )#* (N/K_o) )

            phi_t_0 = torch.cat((phi_t_0_i, phi_t_0_o))

            max_val = phi_t_0[~torch.isnan(phi_t_0)].max()
            phi_t_0[torch.isnan(phi_t_0)] = -max_val
            phi_t_0[~torch.isfinite(phi_t_0)] = - max_val
            phi_t_0_i[K_i == 0] = - max_val
            phi_t_0_o[K_o == 0] = - max_val
        else:
            raise Exception("Not ready yet")
        return self.identify_par_vec(phi_t_0, self.par_vec_id_type)

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        N = self.N
        # Restrict the distribution parameters.
        dist_par_re = self.link_dist_par(dist_par_un, N, A_t=A_t)
        if self.distr == 'gamma':
            EYcond_mat = self.exp_of_fit_plus_reg(phi, beta=beta, X_t=X_t)
           #we already took into account the dimension of dist_par above when restricting it
            rate = torch.div(dist_par_re, EYcond_mat[A_t])
            distr_obj = torch.distributions.gamma.Gamma(dist_par_re, rate)

        elif self.distr == 'lognormal':
            log_EYcond_mat = self.exp_of_fit_plus_reg(phi, beta=beta, X_t=X_t, ret_log=True)
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
      
        if (self.distr == 'gamma') and (self.like_type in [0, 1]):# if non torch computation of the likelihood is required
            # Restrict the distribution parameters.
            dist_par_re = self.link_dist_par(dist_par_un, N, A_t)
            if self.like_type==0:
                """ numerically stable version """
                log_EYcond_mat = self.exp_of_fit_plus_reg(phi, beta=beta, X_t=X_t, ret_log=True)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(log_EYcond_mat[A_t])
                #tmp3 = - torch.sum(torch.div(Y_t[A_t], torch.exp(log_EYcond_mat[A_t])))
                tmp3 = - torch.sum(torch.exp(torch.log(Y_t[A_t])-log_EYcond_mat[A_t] + dist_par_re.log() ))
                out = tmp + tmp1 + tmp2 + tmp3
            elif self.like_type == 1:
                EYcond_mat = self.exp_of_fit_plus_reg(phi, beta=beta, X_t=X_t)
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

        limit_dist_par_re = True
        if limit_dist_par_re:
            dist_par_re = soft_lu_bound(dist_par_re, l_limit=0, u_limit=10)

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
        
        Y_t, X_t = self.get_train_obs_t(t)
        phi_t, _, beta_t = self.get_par_t(t)
        EYcond_mat = self.exp_of_fit_plus_reg(phi_t, beta=beta_t, X_t=X_t)
        EYcond_mat = putZeroDiag(EYcond_mat)
        A_t=putZeroDiag(Y_t)>0
        return torch.sum(Y_t[A_t])/torch.sum(EYcond_mat[A_t]), torch.mean(Y_t[A_t]/EYcond_mat[A_t])

    def get_Y_T_to_save(self):
        return self.Y_T

    def sample_Y_T(self, A_T, avoid_empty=False):
        if A_T.dtype != torch.bool:
            A_T = A_T >0

        return self.sample_Y_T_from_par_list(self.T_all, self.phi_T, X_T = self.X_T, beta_T=self.beta_T, dist_par_un_T=self.dist_par_un_T, avoid_empty=avoid_empty, A_T=A_T)

    def info_filter(self):
        return self.model_class + super().info_filter()


    def score_t(self, t, rescale_flag, phi_flag, dist_par_un_flag, beta_flag):

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
        if  dist_par_un_flag | beta_flag:
            
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t, dist_par_un=dist_par_un_t)
            # TODO Add the analytical gradient for beta and dist_par_un KEEPin MIND the ovflw limit for dist_par_un  
            if beta_flag:
                score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]
            if dist_par_un_flag:
                score_dict["dist_par_un"] = grad(like_t, dist_par_un_t, create_graph=True)[0]
            if rescale_flag:
                pass
                # raise "Rescaling not ready for beta and dist par"
        
        if phi_flag:
            sigma_mat = self.link_dist_par(dist_par_un_t, self.N)
            log_cond_exp_Y, log_cond_exp_Y_restr, cond_exp_Y = self.exp_of_fit_plus_reg(phi_t, beta=beta_t, X_t=X_t, ret_all=True)

            if self.distr == 'gamma':
                tmp = (Y_t.clone()/cond_exp_Y - A_t)*sigma_mat
                if rescale_flag:
                    diag_resc_mat = sigma_mat*A_t

            elif self.distr == 'lognormal':
                sigma2_mat = sigma_mat**2
                log_Y_t = torch.zeros(self.N, self.N)
                log_Y_t[A_t_bool] = Y_t[A_t_bool].log()
                tmp = (log_Y_t - log_cond_exp_Y_restr*A_t)/sigma2_mat + A_t/2
                if rescale_flag:
                    diag_resc_mat = (sigma_mat**2)**A_t

            if self.avoid_ovflw_fun_flag:
                L = self.ovflw_exp_L_limit
                U = self.ovflw_exp_U_limit
                # multiply the score by the derivative of the overflow limit function
                soft_bnd_der_mat = (1 - torch.tanh(2*((log_cond_exp_Y - L)/(L - U)) + 1)**2)
                tmp = soft_bnd_der_mat * tmp
                if rescale_flag:
                    diag_resc_mat = diag_resc_mat * (soft_bnd_der_mat**2)

            score_phi = strIO_from_mat(tmp)

            #rescale score if required
            if rescale_flag:
                diag_resc = strIO_from_mat(diag_resc_mat)
                diag_resc[diag_resc==0] = 1
                score_phi = score_phi/diag_resc.sqrt()

            score_dict["phi"] = score_phi

        return score_dict


    def hess_t(self, t, phi_flag, dist_par_un_flag, beta_flag):

        """
        given the observations and the ZA gamma parameters (i.e. the cond mean
        matrix and the dist_par_un par), return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """

        Y_t, X_t = self.get_obs_t(t)

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        if phi_t is not None:
            phi_t.requires_grad = phi_flag
        if dist_par_un_t is not None:
            dist_par_un_t.requires_grad = dist_par_un_flag
        if beta_t is not None:
            beta_t.requires_grad = beta_flag

        hess_dict = {}
        like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t, dist_par_un=dist_par_un_t)

        if beta_flag:
            fun = lambda x: self.loglike_t(Y_t, phi_t, beta=x, X_t=X_t, dist_par_un=dist_par_un_t)

            hess_dict["beta"] = hessian(fun, beta_t)[0]
        if dist_par_un_flag:
            fun = lambda x: self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t, dist_par_un=x)
            hess_dict["dist_par_un"] = hessian(fun, dist_par_un_t)[0]
        
        if phi_flag:
            fun = lambda x: self.loglike_t(Y_t, x, beta=beta_t, X_t=X_t, dist_par_un=dist_par_un_t)
            hess_dict["phi"] = hessian(fun, phi_t)
        
        return hess_dict

class dirSpW1_SD(dirGraphs_SD, dirSpW1_sequence_ss):


    def __init__(self, Y_T, **kwargs):

        super().__init__(Y_T, **kwargs)

        self.mod_stat = self.get_mod_stat(Y_T, **kwargs)

    

    def get_mod_stat(Y_T, **kwargs):

        kwargs = self.set_all_tv_opt_to_false(kwargs)  
        kwargs["phi_tv"] = False
        kwargs["dist_par_tv"] = False
        beta_stat = copy.deepcopy(self.beta_tv)
        beta_stat = False
        kwargs["beta_tv"] = beta_stat
        kwargs = self.remove_sd_specific_keys(kwargs)

        mod_stat = dirSpW1_sequence_ss(Y_T, **kwargs)
        mod_stat.opt_options_ss_seq["max_opt_iter"] = 5000
        mod_stat.opt_options_ss_seq["disable_logging"] = True
        
        return mod_stat


    def out_of_sample_eval(self, exclude_never_obs_train=True):
 
        if not exclude_never_obs_train:
            raise

        Y_vec_all, F_Y_vec_all = self.get_out_of_sample_obs_and_pred(only_present=True)

        eval_dict = { "mse":mean_squared_error(Y_vec_all, F_Y_vec_all),
            "mse_log":mean_squared_log_error(Y_vec_all, F_Y_vec_all),
            "mae":mean_absolute_error(Y_vec_all, F_Y_vec_all),
            "mae_pct":mean_absolute_percentage_error(Y_vec_all, F_Y_vec_all),
            "r2_score":r2_score(Y_vec_all, F_Y_vec_all)}
        
        return eval_dict


    def info_filter(self):
        return self.model_class + super().info_filter()


# Binary Graphs
class dirBin1_sequence_ss(dirGraphs_sequence_ss):
    
    def __init__(self, Y_T, *args, avoid_ovflw_fun_flag=False, **kwargs):

        Y_T = tens(Y_T > 0)

        super().__init__( Y_T, *args, distr = "bernoulli", **kwargs)
  
        self.obs_type = "bin"

        self.model_class = "dirBin1"

        self.check_mat_is_bin()

    def check_mat_is_bin(self):
        check_mat_pos = 0 <=  self.Y_T
        check_mat_bin = self.Y_T <= 1
        if not check_mat_pos.all().item():
            raise Exception("matrix non positive")
        if not check_mat_bin.all().item():
            raise Exception("matrix non binary")


    def invPiMat(self, phi, beta, X_t, ret_log=False):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(phi_i + phi_j)
        """
        return putZeroDiag(self.exp_of_fit_plus_reg(phi, beta, X_t, ret_log=ret_log, ret_all=False))

    def exp_A(self, phi, beta=None, X_t=None):
        """
        given the vector of unrestricted parameters, compute the expected
        matrix, can have non zero elements on the diagonal!!
        """
        invPiMat_= self.invPiMat(phi, beta=beta, X_t=X_t)
        out = invPiMat_/(1 + invPiMat_)
        return out

    def exp_Y(self, phi, beta, X_t):
        return self.exp_A(phi, beta=beta, X_t=X_t)

    def zero_deg_par_fun(self, Y_t, phi, method, degIO=None):
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
                logger.error(((par_i  - par_i[imin])[diffDegs != 0]) / diffDegs[diffDegs != 0])
                raise
            diffDegs = degO[degO  != 0] - degO[degO != 0][omin]
            avgParStepO = torch.mean((par_o[degO != 0] - par_o[degO != 0][omin])[diffDegs != 0] / diffDegs[diffDegs != 0])
            if not torch.isfinite(avgParStepO):
                logger.error(((par_i - par_i[imin])[diffDegs != 0]) / diffDegs[diffDegs != 0])
                raise
            # subtract a multiple of this average step to fitnesses of smallest node
            zero_deg_par_i = par_i[degI != 0][imin] - avgParStepI * degI[degI != 0][imin]
            zero_deg_par_o = par_o[degO != 0][omin] - avgParStepO * degO[degO != 0][omin]

        elif method == "EXPECTATION":
            #  Set the fitness to a value s.t  the expectation
            # of the degrees for each node with deg zero is small
            N = self.N
            max_prob = tens(0.01)
            if degIO is None:
                degIO = strIO_from_mat(Y_t)
            if (sum(degIO) == 0): #| (sum(phi) == 0):
                raise
            par_i, par_o = splitVec(phi)
            zero_deg_par_i = - par_o.max() + torch.log(max_prob/N)
            zero_deg_par_o = - par_i.max() + torch.log(max_prob/N)

        return zero_deg_par_i, zero_deg_par_o

    def set_zero_deg_par(self, Y_t, phi_in, method="EXPECTATION", degIO=None):
        phi = phi_in.clone()
        if degIO is None:
            degIO = strIO_from_mat(Y_t)
        N = self.N
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
        if self.size_phi_t == 2*self.N:    
            if degIO is None:
                degIO = tens(strIO_from_mat(Y)).squeeze()
            ldeg = degIO.log()
            nnzInds = degIO > 0
            phi_0 = torch.ones(degIO.shape[0]) * (-15)
            phi_0[nnzInds] = degIO[nnzInds].log()
            for i in range(n_iter):
                phi_0 = self.phiFunc(phi_0, ldeg)
                #phi_0[~nnzInds] = -15
            phi_0 = self.set_zero_deg_par(Y, phi_0, degIO=degIO)
        elif self.phi_T is None:
            return None
        else: 
            raise

        return phi_0.clone()

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
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
            dist = self.dist_from_pars(phi, beta, X_t, dist_par_un=None, A_t=None)
            log_probs = dist.log_prob(Y_t).clone()
            out = torch.sum(log_probs)
        return out.clone()

    def check_exp_vals(self, t):
        Y_t, X_t = self.get_train_obs_t(t)
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

    def sample_Y_T(self, avoid_empty=True):
        return self.sample_Y_T_from_par_list(self.T_all, self.phi_T, X_T = self.X_T, beta_T=self.beta_T, dist_par_un_T=self.dist_par_un_T, avoid_empty=avoid_empty)

    def info_filter(self):
        return self.model_class + super().info_filter()

    def eval_prediction(self, Y_tp1, phi, beta, X_tp1):
        pass

    def score_t(self, t, rescale_flag, phi_flag, dist_par_un_flag, beta_flag):
        Y_t, X_t = self.get_obs_t(t)

        phi_t, _, beta_t = self.get_par_t(t)

        A_t = Y_t 

        score_dict = {}
        if  beta_flag:
            
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t)

            if self.any_beta_tv():
                score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]
        
            if rescale_flag:
                pass
                # raise "Rescaling not ready for beta and dist par"

        if phi_flag:

            exp_A = self.exp_A(phi_t, beta=beta_t, X_t=X_t)

            tmp = A_t - exp_A

            if rescale_flag:
                diag_resc_mat = exp_A * (1 - exp_A)

            if self.avoid_ovflw_fun_flag:
                log_inv_pi__mat = self.invPiMat(phi_t, beta=beta_t, X_t=X_t, ret_log=True)
                L = self.ovflw_exp_L_limit
                U = self.ovflw_exp_U_limit
                # multiply the score by the derivative of the overflow limit function
                soft_bnd_der_mat = (1 - torch.tanh(2*((log_inv_pi__mat - L)/(L - U)) + 1)**2)
                tmp = soft_bnd_der_mat * tmp
                if rescale_flag:
                    diag_resc_mat = diag_resc_mat * (soft_bnd_der_mat**2)

            score_phi = strIO_from_mat(tmp)
     
            #rescale score if required
            if rescale_flag:
                diag_resc = torch.cat((diag_resc_mat.sum(dim=0), diag_resc_mat.sum(dim=1)))
                diag_resc[diag_resc==0] = 1
                score_phi = score_phi/diag_resc.sqrt()

            score_dict["phi"] = score_phi
    
        return score_dict

class dirBin1_SD(dirGraphs_SD, dirBin1_sequence_ss):

    def __init__(self, Y_T, **kwargs): 

        Y_T = tens(Y_T > 0)

        super().__init__( Y_T, **kwargs) 

        self.check_mat_is_bin()

        self.mod_stat = self.get_mod_stat(Y_T, **kwargs)


    def get_mod_stat(self, Y_T, **kwargs):
        kwargs["phi_tv"] = False
        kwargs["dist_par_tv"] = False
        beta_stat = copy.deepcopy(self.beta_tv)
        beta_stat = False
        kwargs["beta_tv"] = beta_stat

        kwargs = self.remove_sd_specific_keys(kwargs)

        mod_stat = dirBin1_sequence_ss(Y_T, **kwargs)

        mod_stat.opt_options_ss_seq["max_opt_iter"] = 5000
        mod_stat.opt_options_ss_seq["disable_logging"] = True

        return mod_stat


  
   
   
    def out_of_sample_eval(self, exclude_never_obs_train=True):
        if exclude_never_obs_train:
            inds_keep_subset = self.get_train_Y_T().sum(dim=(2)) > 0
        else:
            inds_keep_subset = None

        Y_vec_all, F_Y_vec_all = self.get_out_of_sample_obs_and_pred(only_present=False, inds_keep_subset=inds_keep_subset)
        logger.info(f"out of sample eval on {Y_vec_all.size} observations")
        auc_score = roc_auc_score(Y_vec_all, F_Y_vec_all)
        return {"auc_score":auc_score}

    def info_filter(self):
        return self.model_class + super().info_filter()

        
# Instantiate function

def get_gen_fit_mod(bin_or_w, ss_or_sd, Y_T, **kwargs):
    if (bin_or_w == "bin") and (ss_or_sd == "ss"):
        return dirBin1_sequence_ss(Y_T, **kwargs)
    elif (bin_or_w == "w") and (ss_or_sd == "ss"):
        return dirSpW1_sequence_ss(Y_T, **kwargs)
    elif (bin_or_w == "bin") and (ss_or_sd == "sd"):
        return dirBin1_SD(Y_T, **kwargs)
    elif (bin_or_w == "w") and (ss_or_sd == "sd"):
        return dirSpW1_SD(Y_T, **kwargs)
    else:
        raise



