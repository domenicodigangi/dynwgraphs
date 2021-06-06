"""

Functions for Zero Augmented models for sparse weighted dynamical networks, coded using pytorch
Author: Domenico Di Gangi


"""
import torch
import sys
import numpy as np
import os
sys.path.append("./src/")
from .utils import splitVec, tens, putZeroDiag, putZeroDiag_T, optim_torch, gen_test_net, soft_lu_bound, soft_l_bound,\
    degIO_from_mat, strIO_from_mat, tic, toc
from torch.autograd import grad
#----------------------------------- Zero Augmented Static model functions
#self = dirSpW1_dynNet_SD(ovflw_lm=True, rescale_SD = False )



class dirSpW1_staNet(object):
    """
    This a class  for directed weighted sparse static networks (or sequences), modelled with a zero augmented distribution, one  parameter per each node (hence the 1 in the name)
    """

    def __init__(self, ovflw_lm=True, distribution='gamma'):
        self.ovflw_lm = ovflw_lm
        self.distr = distribution

        self.ovflw_exp_L_limit = -50
        self.ovflw_exp_U_limit = 40

    @staticmethod
    def regr_product(beta, X_t):
        """
        Given a matrix of regressors and a vector of parameters obtain a matrix that will be sum to the matrix obtained
        from of nodes specific unrestricted dynamical paramters
        """
        if not (X_t.dim() == 3):
            raise
        N = X_t.shape[0]
        if beta.shape[0] == 1: # one parameter for all links
            prod = beta * X_t
        elif beta.shape[0] == N: # one parameter for each node for each regressor
            prod = torch.mul(beta * beta.unsqueeze(1), X_t)
        return prod.sum(dim=2)

    def cond_exp_Y(self, phi, beta=None, X_t=None, p_mat=None, ret_log=False, ret_all=False):
        """
        given the parameters related with the weights, compute the conditional expectation of matrix Y.
        the conditioning is on each matrix element being greater than zero
        can have non zero elements on the diagonal!!
        """
        phi_i, phi_o = splitVec(phi)
        log_Econd_mat = phi_i + phi_o.unsqueeze(1)
        if X_t is not None:
            log_Econd_mat = log_Econd_mat + self.regr_product(beta, X_t)

        if self.ovflw_lm:
            """if required force the exponent to stay within overflow-safe bounds"""
            log_Econd_mat_restr = \
                soft_lu_bound(log_Econd_mat, l_limit=self.ovflw_exp_L_limit, u_limit=self.ovflw_exp_U_limit)
        else:
            log_Econd_mat_restr = log_Econd_mat

        if ret_all:
            return log_Econd_mat, log_Econd_mat_restr, torch.exp(log_Econd_mat_restr)
        elif ret_log:
            return log_Econd_mat_restr #putZeroDiag(log_Econd_mat)
        else:
            return torch.exp(log_Econd_mat_restr)# putZeroDiag(torch.exp(log_Econd_mat))

    @staticmethod
    def identify(phi, id_type=1):
        """ enforce an identification condition on the parameters of ZA gamma net model
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if id_type == 0:
            """set one to zero"""
            d = phi_i[0]
        elif id_type == 1:
            """set in and out difference to zero """
            d = (phi_i.mean() - phi_o.mean())/2

        phi_i_out = phi_i - d
        phi_o_out = phi_o + d
        return torch.cat((phi_i_out, phi_o_out))

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

        return self.identify(phi_t_0)

    def start_phi_from_obs_T(self, Y_T):
        N = Y_T.shape[0]
        T = Y_T.shape[2]
        phi_T = torch.zeros(2*N, T)
        for t in range(T):
            Y_t = Y_T[:, :, t]
            phi_T[:, t] = self.start_phi_from_obs(Y_t)
        return phi_T

    def dist_from_pars(self, distribution, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        N = phi.shape[0]//2
        # Restrict the distribution parameters.
        dist_par_re = self.link_dist_par(dist_par_un, N, A_t=A_t)
        if distribution == 'gamma':
            EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
            if A_t is None:
                rate = torch.div(dist_par_re, EYcond_mat)
                dist = torch.distributions.gamma.Gamma(dist_par_re, rate)
            else:# if A_t is given, we already took into account the dimension of dist_par above when restricting it
                rate = torch.div(dist_par_re, EYcond_mat[A_t])
                dist = torch.distributions.gamma.Gamma(dist_par_re, rate)

        elif distribution == 'lognormal':
            log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
            if A_t is None:
                sigma = dist_par_re
                mu = log_EYcond_mat - (sigma ** 2) / 2
                dist = torch.distributions.log_normal.LogNormal(mu, sigma)
            else:  # if A_t is given, we already took into account the dimension of dist_par above when restricting it
                sigma = dist_par_re
                mu = log_EYcond_mat[A_t] - (sigma ** 2) / 2
                dist = torch.distributions.log_normal.LogNormal(mu, sigma)
        return dist

    def loglike_t(self, Y_t, phi, beta=None, X_t=None, dist_par_un=None, like_type=2):
        """ The log likelihood of the zero augmented gamma network model as a funciton of the
        observations and the matrices of parameters p_mat (pi = p/(1+p), where pi is the prob of being
        nnz), and EYcond_mat E[Y_ij |Y_ij > 0 ]
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t)
        A_t = Y_t > 0
        N = A_t.shape[0]
        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val()

        if (self.distr == 'gamma') and (like_type in [0, 1]):# if non torch computation of the likelihood is required
            # Restrict the distribution parameters.
            dist_par_re = self.link_dist_par(dist_par_un, N, A_t)
            if like_type==0:
                """ numerically stable version """
                log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(log_EYcond_mat[A_t])
                #tmp3 = - torch.sum(torch.div(Y_t[A_t], torch.exp(log_EYcond_mat[A_t])))
                tmp3 = - torch.sum(torch.exp(torch.log(Y_t[A_t])-log_EYcond_mat[A_t] + dist_par_re.log() ))
                out = tmp + tmp1 + tmp2 + tmp3
            elif like_type == 1:
                EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(torch.log(EYcond_mat[A_t]))
                tmp3 = - torch.sum(torch.div(Y_t[A_t], EYcond_mat[A_t])*dist_par_re)
                out = tmp + tmp1 + tmp2 + tmp3

        else:# compute the likelihood using torch buit in functions
            dist = self.dist_from_pars(self.distr, phi, beta, X_t, dist_par_un, A_t=A_t)
            log_probs = dist.log_prob(Y_t[A_t])
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

    def dist_par_un_start_val(self, dim_dist_par_un=1):
        if self.distr=='gamma':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(dim_dist_par_un)
        elif self.distr=='lognormal':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(dim_dist_par_un)
        return dist_par_un0


    def estimate_ss_t(self, Y_t, X_t=None, beta_t=None, phi_0=None, dist_par_un_t=None, like_type=2,
                        est_dist_par=False, dim_dist_par_un=1, est_beta=False, dim_beta = 1, min_n_iter=200,
                        opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False, print_every=10):
        """
        single snapshot Maximum logLikelihood estimate of phi_t, and if not given also of dist_par_un_t and  beta_t
        """

        N = Y_t.shape[0]

        # Define starting values, if an inut is given but the parameter has to be estimateduse input as starting value
        # starting value for phi_t
        if phi_0 is None:
            phi_0 = self.start_phi_from_obs(Y_t)
        unPar_0 = phi_0#.clone().detach()
        # if no value for dist_par is provided, set a starting value and estimate it
        if est_dist_par:
            N_dis_par = dim_dist_par_un
            # if given use input as starting point
            if dist_par_un_t is None:
                dist_par_un_0 = self.dist_par_un_start_val(dim_dist_par_un)
            else:
                dist_par_un_0 = dist_par_un_t#.clone().detach()
            unPar_0 = torch.cat((unPar_0, dist_par_un_0))
        if est_beta:
            #if given use beta_t as starting point else start at zero
            if beta_t is None:
                n_reg = X_t.shape[2]
                beta_0 = torch.randn(dim_beta, n_reg)*0.01
            else:
                beta_0 = beta_t#.clone().detach()
            unPar_0 = torch.cat((unPar_0, beta_0.view(-1)))

        if (not est_beta) and (not est_dist_par):  # estimate only phi_t
            def obj_fun(unPar):
                return - self.loglike_t(Y_t, unPar, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t,
                                          like_type=like_type)

            all_par_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                            min_n_iter=min_n_iter,
                                            plot_flag=plot_flag, print_flag=print_flag, print_every=print_every)

        elif (not est_beta) and (est_dist_par): # estimate  phi_t and dist_par
            def obj_fun(unPar):
                return - self.loglike_t(Y_t, unPar[:2*N], X_t=X_t, beta=beta_t, dist_par_un=unPar[2*N:],
                                          like_type=like_type)

            all_par_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                            min_n_iter=min_n_iter,
                                            plot_flag=plot_flag, print_flag=print_flag, print_every=print_every)

        elif (est_beta) and (not est_dist_par): # estimate phi_t dist_par and and beta
            def obj_fun(unPar):
                return - self.loglike_t(Y_t, unPar[:2*N], X_t=X_t,
                                          beta=unPar[2*N:].view(dim_beta, n_reg),
                                          dist_par_un=None,
                                          like_type=like_type)

            all_par_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                            min_n_iter=min_n_iter,
                                            plot_flag=plot_flag, print_flag=print_flag, print_every=print_every)

        elif (est_beta) and (est_dist_par): # estimate phi_t dist_par and and beta
            def obj_fun(unPar):
                return - self.loglike_t(Y_t, unPar[:2*N], X_t=X_t,
                                          beta=unPar[2*N + N_dis_par:].view(dim_beta, n_reg),
                                          dist_par_un=unPar[2*N:2*N + N_dis_par],
                                          like_type=like_type)

            all_par_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                            min_n_iter=min_n_iter,
                                            plot_flag=plot_flag, print_flag=print_flag, print_every=print_every)

        #Identify the phi_t part
        all_par_est[:2*N] = self.identify(all_par_est[:2*N])

        return all_par_est, diag


    def ss_filt(self, Y_T, X_T=None, beta=None, phi_T_0=None, dist_par_un=None, like_type=2,
                est_dist_par=False, dim_dist_par_un=1, est_beta=False, dim_beta = 1,
                opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False, print_every=10,
                rel_improv_tol=5*1e-7, no_improv_max_count=30,
                min_n_iter=150, bandwidth=50, small_grad_th=1e-6):

        """
        Single snapsot sequence filter.
        return the sequence  of single snapshot estimates of phi and if required also of dist_par and beta
        """
        T = Y_T.shape[2]
        N = Y_T.shape[0]
        #always estimate phi_T
        all_par_est_T = torch.zeros(2 * N, T)

        if est_dist_par:
            all_par_est_T = torch.cat((all_par_est_T, torch.zeros(dim_dist_par_un, T) ), dim=1)
        if est_beta:
            n_reg = X_T.shape[2]
            all_par_est_T = torch.cat((all_par_est_T, torch.zeros(dim_beta*n_reg, T)), dim=1)

        diag_T = []
        X_t = None
        for t in range(T):
            # estimate each phi_t given dist_par_un
            Y_t = Y_T[:, :, t]

            if phi_T_0 is None:
                phi_t_0 = None
            else:
                phi_t_0 = phi_T_0[:, t]

            if X_T is not None:
                X_t = X_T[:, :, :, t]

            all_par_t, diag_t = self.estimate_ss_t(Y_t,
                                                     phi_0=phi_t_0,
                                                     est_dist_par=est_dist_par,
                                                     dist_par_un_t=dist_par_un, dim_dist_par_un=dim_dist_par_un,
                                                     est_beta=est_beta,
                                                     dim_beta=dim_beta, beta_t=beta, X_t=X_t,
                                                     opt_steps=opt_steps,
                                                     like_type=like_type,
                                                     lRate=lRate, opt_n=opt_n,
                                                     print_flag=print_flag, plot_flag=plot_flag,
                                                     print_every=print_every, min_n_iter=min_n_iter)

            all_par_est_T[:, t] = all_par_t
            diag_T.append(diag_t)
            # print(all_par_t.mean())
            # print(t)
        return all_par_est_T, diag_T


    def estimate_dist_par_const_given_phi_T(self, Y_T, phi_T, dim_dist_par_un, X_T=None, beta=None,
                                            dist_par_un_0=None, like_type=2,
                                            opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False,
                                            print_every=10,
                                            rel_improv_tol=5*1e-7, no_improv_max_count=30,
                                            min_n_iter=150, bandwidth=50, small_grad_th=1e-6):

        """single snapshot Maximum logLikelihood estimate given beta_t"""

        N = Y_T.shape[0]
        T = Y_T.shape[2]
        if dist_par_un_0 is None:
            dist_par_un_0 = torch.zeros(dim_dist_par_un)

        unPar_0 = dist_par_un_0#.clone().detach()

        # the likelihood for beta_t is the sum of all the single snapshots likelihoods, given phi_T
        def obj_fun(unPar):
            logl_T=0
            X_t=None
            for t in range(T):
                if X_T is not None:
                    X_t=X_T[:, :, :, t]
                logl_T = logl_T + self.loglike_t(Y_T[:, :, t], phi_T[:, t], X_t=X_t, beta=beta,
                                                   dist_par_un=unPar, like_type=like_type)
            return - logl_T

        dist_par_un_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                            plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                            print_fun=lambda x: torch.exp(x),
                                            rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                            min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)

        return dist_par_un_est, diag


    def estimate_beta_const_given_phi_T(self, Y_T, X_T, phi_T, dim_beta, dist_par_un,  beta_0=None, like_type=2,
                                        opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False,
                                        print_every=10,
                                        rel_improv_tol=5*1e-7, no_improv_max_count=30,
                                        min_n_iter=150, bandwidth=50, small_grad_th=1e-6):

        """single snapshot Maximum logLikelihood estimate given beta_t"""

        T = Y_T.shape[2]
        n_reg = X_T.shape[2]
        if beta_0 is None:
            n_reg = X_T.shape[2]
            beta_0 = torch.randn(dim_beta, n_reg)*0.01

        unPar_0 = beta_0.view(-1).clone().detach()

        # the likelihood for beta_t is the sum of all the single snapshots likelihoods, given phi_T
        def obj_fun(unPar):
            logl_T=0
            for t in range(T):
                logl_T = logl_T + self.loglike_t(Y_T[:, :, t], phi_T[:, t], X_t=X_T[:, :, :, t],
                                                   beta=unPar.view(dim_beta, n_reg),
                                                   dist_par_un=dist_par_un, like_type=like_type)
            return - logl_T

        beta_t_est_flat, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                       plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                        rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                        min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)

        beta_t_est = beta_t_est_flat.view(dim_beta, n_reg)
        return beta_t_est, diag


    def ss_filt_est_beta_dist_par_const(self, Y_T, X_T=None, beta=None, phi_T=None, dist_par_un=None, like_type=2,
                                          est_const_dist_par=False, dim_dist_par_un=1,
                                          est_const_beta=False, dim_beta = 1,
                                          opt_large_steps=10, opt_n=1,   opt_steps_phi=500, lRate_phi=0.01,
                                          opt_steps_dist_par=500, lRate_dist_par=0.01,
                                          opt_steps_beta=400, lRate_beta=0.01,
                                          print_flag_phi=False, print_flag_dist_par=False, print_flag_beta=False,
                                          print_every=50,
                                          rel_improv_tol=5*1e-7, no_improv_max_count=30,
                                          min_n_iter=150, bandwidth=50, small_grad_th=1e-6):
        """
        Static sequence filter.
         return the sequence  of single snapshot estimates for phi and the corresponding estimate for beta_t
          to avoid one very large optimization alternate:
            1 estimate of sequences of  phi given beta_t
            2 estimate of beta_t given phi_T
        """
        T = Y_T.shape[2]
        N = Y_T.shape[0]

        # if starting values or input values are not given, define them
        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val(dim_dist_par_un)
        if beta is None:
            if X_T is not None:
                n_reg = X_T.shape[2]
                beta = torch.zeros(dim_beta, n_reg)
            else:
                beta = torch.zeros(1)
        if phi_T is None:
            phi_T = self.start_phi_from_obs_T(Y_T)

        diag = []
        for n in range(opt_large_steps+1):
            #print(n)

            phi_T, diag_phi_t = self.ss_filt(Y_T, X_T=X_T, beta=beta, phi_T_0=phi_T,
                                            dist_par_un=dist_par_un,
                                            like_type=like_type, est_dist_par=False, dim_dist_par_un=dim_dist_par_un,
                                            est_beta=False, dim_beta=dim_beta,
                                            opt_steps=opt_steps_phi, opt_n=opt_n,
                                            lRate=lRate_phi, print_flag=print_flag_phi, print_every=print_every,
                                            rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                            min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)


            diag.append(self.like_seq(Y_T, phi_T, dist_par_un_T=dist_par_un, X_T=X_T, beta=beta).item())
            # print(phi_T)
            # estimate beta_t given phi_T and dist_par
            if est_const_beta:
                beta, diag_beta = self.estimate_beta_const_given_phi_T(Y_T, X_T, phi_T.clone().detach(), dim_beta,
                                                                        dist_par_un=dist_par_un.clone().detach(),
                                                                        beta_0=beta.clone().detach(),
                                                                        lRate=lRate_beta,     opt_steps=opt_steps_beta,
                                                                        print_flag=print_flag_beta,
                                                                        print_every=print_every,
                                                                        rel_improv_tol=rel_improv_tol,
                                                                        no_improv_max_count=no_improv_max_count,
                                                                        min_n_iter=min_n_iter, bandwidth=bandwidth,
                                                                        small_grad_th=small_grad_th)

                diag.append(self.like_seq(Y_T, phi_T, dist_par_un_T=dist_par_un, X_T=X_T, beta=beta).item())

            # print(beta)

            # estimate dist_par given phi_T and beta_t
            if est_const_dist_par:
                dist_par_un, diag_dist_par_un = self.estimate_dist_par_const_given_phi_T(Y_T, phi_T.clone().detach(),
                                                                            dim_dist_par_un,
                                                                            X_T=X_T, beta=beta.clone().detach(),
                                                                            dist_par_un_0=dist_par_un,
                                                                            lRate=lRate_dist_par,
                                                                            opt_steps=opt_steps_dist_par,
                                                                            print_flag=print_flag_dist_par,
                                                                            print_every=print_every,
                                                                            rel_improv_tol=rel_improv_tol,
                                                                            no_improv_max_count=no_improv_max_count,
                                                                            min_n_iter=min_n_iter,
                                                                            bandwidth=bandwidth,
                                                                            small_grad_th=small_grad_th)

                diag.append(self.like_seq(Y_T, phi_T, dist_par_un_T=dist_par_un, X_T=X_T, beta=beta).item())
            # print(dist_par_un)

        return phi_T.clone().detach(), dist_par_un.clone().detach(), beta.clone().detach(), diag

    def like_seq(self, Y_T, phi_T, dist_par_un_T, X_T=None, beta=None):
        T = Y_T.shape[2]
        if (len(dist_par_un_T.shape)==1):
            dist_par_un_T = dist_par_un_T.unsqueeze(1).repeat_interleave(T, dim=1)
        elif (dist_par_un_T.shape[1] ==1):
            dist_par_un_T = dist_par_un_T.repeat_interleave(T, dim=1)
        like_seq = 0
        X_t = None
        for t in range(T):
            # estimate each phi_t given beta_t
            Y_t = Y_T[:, :, t]
            if X_T is not None:
                X_t = X_T[:, :, :, t]
            like_seq = like_seq + self.loglike_t(Y_t, phi_T[:, t], dist_par_un=dist_par_un_T[:, t], beta=beta, X_t=X_t)
        return like_seq

    def check_tot_exp(self, Y_t, phi_t, X_t=None, beta_t=None):
        EYcond_mat = self.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t)
        EYcond_mat = putZeroDiag(EYcond_mat)
        A_t=putZeroDiag(Y_t)>0
        return torch.sum(Y_t[A_t])/torch.sum(EYcond_mat[A_t]), torch.mean(Y_t[A_t]/EYcond_mat[A_t])

    def check_tot_exp_seq(self, Y_T, phi_T, X_T=None, beta_t=None):
        T = Y_T.shape[2]
        ratios = []
        X_t = None
        for t in range(T):
            # estimate each phi_t given beta_t
            Y_t = Y_T[:, :, t]
            if X_T is not None:
                X_t = X_T[:, :, :, t]
            phi_t = phi_T[:, t]
            ratios.append(self.check_tot_exp(Y_t, phi_t, beta_t, X_t))
        return ratios

    def exp_seq(self, phi_T, X_T=None, beta_const=None):
        T = phi_T.shape[1]
        N = phi_T.shape[0]//2
        E_Y_T = torch.zeros(N, N, T)
        X_t = None
        for t in range(T):
            # estimate each phi_t given beta_t
            if X_T is not None:
                X_t = X_T[:, :, :, t]
            phi_t = phi_T[:, t]
            E_Y_T[:, :, t] = self.cond_exp_Y(phi_t, beta=beta_const, X_t=X_t)
        return E_Y_T


    def sample_from_dgps(self, N, T, N_sample, p_T=torch.ones(1, 1, 1)*0.1, dgp_type='sin',
                           n_reg=2, n_reg_beta_tv=1, dim_beta=1, dim_dist_par_un=1, dist_par_un=None,
                           distribution='gamma', Y_0=None, X_T=None):
        torch.manual_seed(2)
        phi_T = torch.zeros(N, T)
        beta_T = torch.zeros(N, T)
        if n_reg>0:
            if X_T is None:
                X_T = torch.distributions.normal.Normal(0.0, 1.0).sample((N, N, n_reg, T))
            X_0 = X_T[:, :, :, 0]
        if Y_0 is None:
            # start from a network with heterogeneous distribution of weights.
            Y_0 = torch.distributions.pareto.Pareto(1.0, 0.8).sample((N, N))
            # randomly set to zero according to p_T[:, :, 0]
            A_0 = torch.distributions.bernoulli.Bernoulli(p_T[:, :, 0]).sample((1,)).view(N, N) == 1
            Y_0[~A_0] = 0

        # estimate static model on the starting network and have phi tv around those values
        all_par_0, diag = self.estimate_ss_t(Y_t=Y_0, est_beta=False,
                                               est_dist_par=True,
                                               dim_dist_par_un=dim_dist_par_un,
                                               dim_beta=dim_beta,
                                               print_flag=False)

        um_phi = all_par_0[:2*N]

        if dist_par_un is None:
            dist_par_un = all_par_0[2*N:2*N + dim_dist_par_un]

        if dgp_type == 'sin':
            period = T
            ampl = 0.3 #* torch.randn((2*N, 1)).abs()
            phi_T = um_phi.unsqueeze(1).detach() + torch.sin(6.28*tens(range(T))/period + torch.randn((2*N, 1))) * ampl

            if n_reg>0:
                beta_T = torch.ones(dim_beta*n_reg, T)
            if n_reg_beta_tv>0:
                ampl_beta = 0.3
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                beta_T[:n_tv_beta_par, :] = beta_T[:n_tv_beta_par, :] + \
                                            torch.sin(6.28*tens(range(T))/period + torch.randn((n_tv_beta_par, 1))) * \
                                            ampl_beta

        if dgp_type == 'step':
            ampl = 0.3  # * torch.randn((2*N, 1)).abs()x
            phi_T = um_phi.unsqueeze(1).detach() + torch.zeros((1, T))
            phi_T[:, T//2:] = phi_T[:, T//2:] + ampl
            if n_reg > 0:
                beta_T = torch.ones(dim_beta * n_reg, T)
            if n_reg_beta_tv > 0:
                ampl_beta = 0.3
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                beta_T[:n_tv_beta_par, :] = beta_T[:n_tv_beta_par, T//2:] + ampl_beta

        if dgp_type == 'ar1':
            ampl = 0.1
            # * torch.randn((2*N, 1)).abs()
            B = 0.99
            w = (1-B)*um_phi.detach()
            # add noises
            phi_T = torch.randn((2*N, T)) * ampl
            phi_T[:, 0] = um_phi.detach()
            for t in range(1, T):
                phi_T[:, t] = w + B * phi_T[:, t-1] + phi_T[:, t]

            if n_reg > 0:
                beta_T = torch.ones(dim_beta * n_reg, T)
            if n_reg_beta_tv > 0:
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                w = (1 - B) * beta_T[:n_tv_beta_par, 0]
                beta_T[:n_tv_beta_par, 1:] = torch.randn((n_tv_beta_par, T-1)) * ampl
                for t in range(1, T):
                    beta_T[:n_tv_beta_par, t] == w + B * beta_T[:n_tv_beta_par, t-1] + beta_T[:n_tv_beta_par, t]

        if False:
            plt.plot(phi_T.transpose(0, 1))
        if False:
            plt.plot(beta_T.transpose(0, 1))

        #smaple given phi_T and p_T and identify phi_T

        Y_T = torch.zeros(N, N, T, N_sample)
        for t in range(T):
            phi = self.identify(phi_T[:, t])
            phi_T[:, t] = phi
            beta = beta_T[:, t].view(dim_beta, n_reg)
            X_t = X_T[:, :, :, t]
            dist_par_re = self.link_dist_par(dist_par_un, N)

            dist = self.dist_from_pars(distribution, phi, beta, X_t, dist_par_re)

            Y_t_S = dist.sample((N_sample,)).permute(1, 2, 0)
            A_t_S = torch.distributions.bernoulli.Bernoulli(p_T[:, :, t]).sample((N_sample, )).view(N, N, N_sample) == 1
            Y_t_S[~A_t_S] = 0
            Y_t_S = putZeroDiag_T(Y_t_S)
            Y_T[:, :, t, :] = Y_t_S

        return Y_T, phi_T, X_T, beta_T, dist_par_un


class dirSpW1_dynNet_SD(dirSpW1_staNet):
    """
        This a class  for directed weighted sparse dynamical networks, modelled with a zero augmented distribution,
        one dynamical parameter per each node (hence the 1 in the name) each evolving with a Score Driven
        Dynamics, and one regressor for each link. the number of parameters associated with the regressors
        will be flexible but for the moment we have a single parameter equal for all links
        """

    def __init__(self, ovflw_lm=False, distribution='gamma', rescale_SD=False, backprop_sd=False):
        dirSpW1_staNet.__init__(self, ovflw_lm=ovflw_lm, distribution=distribution)
        self.rescale_SD = rescale_SD
        self.n_reg_beta_tv = 0
        self.dim_beta = 1
        self.backprop_sd = backprop_sd


    @staticmethod
    def un2re_BA_par( BA_un):
        B_un, A_un = splitVec(BA_un)
        exp_B = torch.exp(B_un)
        return torch.cat((torch.div(exp_B, (1 + exp_B)), torch.exp(A_un)))

    @staticmethod
    def re2un_BA_par(BA_re):
        B_re, A_re = splitVec(BA_re)
        return torch.cat((torch.log(torch.div(B_re, 1 - B_re)), torch.log(A_re)))

    def score_t(self, Y_t, phi_t, beta_t=None, X_t=None, dist_par_un=torch.zeros(1), backprop_score=False, like_type=2):
        """
        given the observations and the ZA gamma parameters (i.e. the cond mean
        matrix and the dist_par_un par), return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """
        N = phi_t.shape[0]//2
        A_t_bool = Y_t > 0
        A_t = tens(A_t_bool)
        if dist_par_un is not None:
            dist_par_un = dist_par_un#.clone()#.detach()

        if beta_t is not None:
            beta = beta_t#.clone().detach()
            beta.requires_grad = True
        else:
            beta = beta_t
        if backprop_score:
            phi = phi_t#.clone().detach()  # is the detach hebeta_tre harmful for the final estimation of SD static pars?????????
            phi.requires_grad = True
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi, beta=beta, X_t=X_t, dist_par_un=dist_par_un, like_type=like_type)
            score_phi = grad(like_t, phi, create_graph=True)[0]
        else:
            phi = phi_t#.clone()#.detach()
            sigma_mat = self.link_dist_par(dist_par_un, N)
            log_cond_exp_Y, log_cond_exp_Y_restr, cond_exp_Y = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_all=True)

            if self.distr == 'gamma':
                tmp = (Y_t.clone()/cond_exp_Y - A_t)*sigma_mat
                if self.rescale_SD:
                    diag_resc_mat = sigma_mat*A_t

            elif self.distr == 'lognormal':
                sigma2_mat = sigma_mat**2
                log_Y_t = torch.zeros(N, N)
                log_Y_t[A_t_bool] = Y_t[A_t_bool].log()
                tmp = (log_Y_t - log_cond_exp_Y_restr*A_t)/sigma2_mat + A_t/2
                if self.rescale_SD:
                    diag_resc_mat = (sigma_mat**2)**A_t

            if self.ovflw_lm:
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

        score_i, score_o = splitVec(score_phi)
        score_beta = None
        if beta_t is not None:
            if self.n_reg_beta_tv > 0:
                score_beta = grad(like_t, beta, create_graph=True)[0][:, :self.n_reg_beta_tv].view(-1)

        return score_i, score_o, score_beta

    def splitBA_init(self, w, B, A, init=False, Y_0=None, X_0=None):
        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        N = (w.shape[0] - N_beta_tv)//2
        N_BA = A.shape[0] - N_beta_tv
        w_i, w_o = splitVec(w[:2*N])
        B_i, B_o = splitVec(B[:2*N_BA])
        A_i, A_o = splitVec(A[:2*N_BA])
        #the last part of the w, B A vectors are the gas parameters for the betas that are tv
        if N_beta_tv > 0:
            w_beta = w[-N_beta_tv:]
            B_beta = B[-N_beta_tv:]
            A_beta = A[-N_beta_tv:]
        else:
            w_beta = None
            B_beta = None
            A_beta = None

        if init:
            if Y_0 is None:
                phi_i_0 = torch.div(w_i, (1 - B_i))
                phi_o_0 = torch.div(w_o, (1 - B_o))
                phi_0 = self.identify(torch.cat((phi_i_0, phi_o_0)))
            else:
                est_beta = False
                if X_0 is not None:
                    est_beta = True
                if self.distr == 'bernoulli':
                    est_dist_par = False
                else:
                    est_dist_par = True
                par_0, diag = self.estimate_ss_t(Y_0, X_t=X_0, beta_t=None, phi_0=None, dist_par_un_t=None,
                                  est_dist_par=est_dist_par, dim_dist_par_un=1, est_beta=est_beta, dim_beta=1, min_n_iter=250,
                                  opt_steps=1500, opt_n=1, lRate=0.01, print_flag=False)

                phi_0 = par_0[:2*N]

            # initialize beta to its unconditional mean (no better way for the moment)
            if N_beta_tv > 0:
                beta_0 = w_beta/(1-B_beta)
                sd_par_0 = torch.cat((phi_0, beta_0))
            else:
                sd_par_0 = phi_0
            return sd_par_0
        else:
            return w_i, w_o, B_i, B_o, A_i, A_o, w_beta, B_beta, A_beta

    def update_dynw_par(self, Y_t, par_sd_t, w, B, A, beta_const=None, X_t = None, dist_par_un=torch.zeros(1)):
        """
        score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
        """
        N = Y_t.shape[0]
        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        phi_t = self.identify(par_sd_t[:2*N])
        phi_i_t, phi_o_t = splitVec(phi_t)
        if N_beta_tv>0:
            beta_t = torch.cat((par_sd_t[-N_beta_tv:].view(self.dim_beta, self.n_reg_beta_tv), beta_const), dim=1)
        else:
            beta_t = beta_const

        s_i, s_o, s_beta = self.score_t(Y_t, phi_t, beta_t=beta_t, X_t=X_t, dist_par_un=dist_par_un,
                                        backprop_score=self.backprop_sd)

        w_i, w_o, B_i, B_o, A_i, A_o, w_beta, B_beta, A_beta = self.splitBA_init(w, B, A)

        phi_i_tp1 = w_i + torch.mul(B_i, phi_i_t) + torch.mul(A_i, s_i)
        phi_o_tp1 = w_o + torch.mul(B_o, phi_o_t) + torch.mul(A_o, s_o)
        phi_tp1 = self.identify(torch.cat((phi_i_tp1, phi_o_tp1)))

        if N_beta_tv>0:
            #s_beta is the score wrt to all betas, here we need only that wrt tv betas (first elements)
            beta_sd_tp1 = w_beta + B_beta * par_sd_t[-N_beta_tv:] + A_beta * s_beta
            sd_par_tp1 = torch.cat((phi_tp1, beta_sd_tp1), dim=0)
            s_all = torch.cat((s_i, s_o, s_beta))
        else:
            sd_par_tp1 = phi_tp1
            s_all = torch.cat((s_i, s_o))
        return sd_par_tp1, s_all

    def sd_dgp(self, w, B, A, p_T, beta_const=None, X_T=None, dist_par_un=torch.zeros(1), N=None, T=None):
        """given the static parameters, sample the dgp with a score driven dynamics.
        equal to the filter except sample the observations for each t.
        """

        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        if p_T is not None:
            N = p_T.shape[0]
            T = p_T.shape[2]
        else:
            p_t = None
        Y_T = torch.zeros((N, N, T))
        dist_par_re = self.link_dist_par(dist_par_un, N)
        sd_par_T = torch.zeros((2 * N + N_beta_tv, 1))

        # initialize the time varying parameters to the unconditional mean
        sd_par_t = self.splitBA_init(w, B, A, init=True)
        X_t = None
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            sd_par_T = torch.cat((sd_par_T, sd_par_t.unsqueeze(1)), 1)

            if X_T is not None:
                X_t = X_T[:, :, :, t]
            #-----------------------------------
            # The following lines must be the only diff in the loop between dgp and
            phi_t = sd_par_t[:2*N]
            if N_beta_tv > 0:
                beta_t = torch.cat((sd_par_t[-N_beta_tv:].view(self.dim_beta, self.n_reg_beta_tv), beta_const), dim=1)
            else:
                beta_t = beta_const
            dist = self.dist_from_pars(self.distr, phi_t, beta_t, X_t, dist_par_re)
            Y_t = dist.sample()
            if p_T is not None:
                p_t = p_T[:, :, t]
                A_t = torch.distributions.bernoulli.Bernoulli(p_t).sample() == 1
                Y_t[~A_t] = 0
            Y_t = putZeroDiag(Y_t)
            Y_T[:, :, t] = Y_t.clone()
            #----------------------------------

            sd_par_tp1, score = self.update_dynw_par(Y_t, sd_par_t, w, B, A,
                                                     beta_const=beta_const, X_t=X_t, dist_par_un=dist_par_un)
            sd_par_t = sd_par_tp1.clone()
            # print(torch.isnan(phi_tp1).sum())

        phi_T = sd_par_T[:2 * N, 1:]
        if N_beta_tv > 0:
            beta_sd_T = sd_par_T[-N_beta_tv:, 1:].view(self.dim_beta, self.n_reg_beta_tv, T)
        else:
            beta_sd_T = None

        return phi_T, beta_sd_T, Y_T

    def sd_filt(self, w, B, A, Y_T, beta_const=None, X_T=None,
                sd_par_0=None, dist_par_un=torch.zeros(1)):
        """given the static parameters and the observations, fiter the dynamical
        parameters with  score driven dynamics.
        """
        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        N = Y_T.shape[0]
        T = Y_T.shape[2]

        sd_par_T = torch.zeros((2 * N + N_beta_tv, 1))

        #if initial value is not provided,  initialize the time varying parameters to the unconditional mean
        if sd_par_0 is None:
            sd_par_t = self.splitBA_init(w, B, A, init=True)
        else:
            sd_par_t = sd_par_0
        X_t = None
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            sd_par_T = torch.cat((sd_par_T, sd_par_t.unsqueeze(1)), 1)

            if X_T is not None:
                X_t = X_T[:, :, :, t]
            # The following line must be the only diff in the loop between dgp and
            # filt func
            Y_t = Y_T[:, :, t]

            sd_par_tp1, score = self.update_dynw_par(Y_t, sd_par_t, w, B, A,
                                                     beta_const=beta_const, X_t=X_t, dist_par_un=dist_par_un)
            sd_par_t = sd_par_tp1.clone()
            #print(torch.isnan(phi_tp1).sum())
            #print(score.sort()[0][:10])
            #print(score.sort()[0][-10:])
        phi_T = sd_par_T[:2*N, 1:]
        if N_beta_tv > 0:
            beta_sd_T = sd_par_T[-N_beta_tv:, 1:].view(self.dim_beta, self.n_reg_beta_tv, T)
        else:
            beta_sd_T = None

        return phi_T, beta_sd_T

    def loglike_sd_filt(self, w, B, A, Y_T, beta_const=None, X_T=None,
                        sd_par_0=None, dist_par_un=torch.zeros(1)):
        """ the loglikelihood of a sd filter for the parameters driving the
        conditional mean, as a function of the static score driven dynamics
        parameters and dist_par_un
        """
        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        phi_T, beta_sd_T = self.sd_filt(w, B, A, Y_T, beta_const=beta_const, X_T=X_T,
                                        sd_par_0=sd_par_0, dist_par_un=dist_par_un)
        T = phi_T.shape[1]
        logl_T = 0
        X_t = None
        beta_t = None
        for t in range(T):
            if X_T is not None:
                X_t = X_T[:, :, :, t]
                if self.n_reg_beta_tv > 0:
                    beta_t = torch.cat((beta_sd_T[:, :, t], beta_const), dim=1)
                else:
                    beta_t = beta_const

           # print(beta_t.shape)
            Y_t = Y_T[:, :, t]
            phi_t = phi_T[:, t]

            logl_T = logl_T + self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t,  dist_par_un=dist_par_un)
        return logl_T

    def seq_bin(self, p_t, Y_T=None, T=None):
        if (T is None) and (Y_T is None):
            raise Exception("seq must be used as a dgp or as a filter")
        return p_t.unsqueeze(2).repeat(1, 1, T)

    def estimate_SD(self, Y_T, opt_n=1, opt_steps=800, lRate=0.005, plot_flag=False, print_flag=False,
                    B0=None, A0=None, W0=None, dist_par_un=None, dim_dist_par_un=1, est_dis_par_un=False,
                    sd_par_0=None, init_filt_um=False,
                    print_every=200, rel_improv_tol=1e-8, no_improv_max_count=30,
                                      min_n_iter=750, bandwidth=250, small_grad_th=1e-6):

        N = Y_T.shape[0]
        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val(dim_dist_par_un)
        if B0 is None:
            B, A = torch.tensor([0.7, 0.7]), torch.ones(2) * 0.0001
            wI, wO = 1 + torch.randn(N), torch.randn(N)
            w = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
        else:
            B = B0
            A = A0
            w = W0
        # initialize the filter at the unconditional mean?
        if init_filt_um:
            sd_par_0=None
        else:
            # use provided initialization or initialize at cross sectional estimate on the first obervation
            if sd_par_0 is None:
                sd_par_0 = self.splitBA_init(w, B, A, init=True, Y_0=Y_T[:, :, 0])

        n = 2 * N
        n_A = n_B = B.shape[0]

        if est_dis_par_un: #estimate the distributions parameters and used input as starting point
            unPar_0 = torch.cat((torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))), dist_par_un))#.clone().detach()
            def obj_fun(unPar):
                reBA = self.un2re_BA_par(unPar[n: n + n_B + n_A])
                return - self.loglike_sd_filt(unPar[:n], reBA[:n_B], reBA[n_B:n_B + n_A], Y_T,
                                                sd_par_0=sd_par_0, dist_par_un=unPar[n + n_B + n_A:])

        else:
            unPar_0 = torch.cat((w, self.re2un_BA_par(torch.cat((B, A)))))#.clone().detach()
            def obj_fun(unPar):
                reBA = self.un2re_BA_par(unPar[n: n + n_B + n_A])
                return - self.loglike_sd_filt(unPar[:n], reBA[:n_B], reBA[n_B:n_B + n_A], Y_T,
                                                sd_par_0=sd_par_0, dist_par_un=dist_par_un)

        unPar_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                      plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                      rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                      min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)

        w_est = unPar_est[:n].clone()
        re_BA_est = self.un2re_BA_par(unPar_est[n: n + n_B + n_A])

        if est_dis_par_un:
            dist_par_un_est = unPar_est[ n + n_B + n_A:].clone()
        else:
            dist_par_un_est = dist_par_un
        return w_est.clone(), re_BA_est[:n_B].clone(), re_BA_est[n_B:].clone(), dist_par_un_est, sd_par_0.clone(), diag

    def estimate_SD_X0(self, Y_T, X_T, dim_beta=None, n_beta_tv=None, dim_dist_par_un=1,
                       sd_par_0=None, init_filt_um=False,
                        opt_n=1, opt_steps=800, lRate=0.005,
                        plot_flag=False, print_flag=False, print_every=10,
                        B0=None, A0=None, W0=None, beta_const_0=None, est_beta=False,  dist_par_un=None,
                       est_dis_par_un=False, rel_improv_tol=1e-8, no_improv_max_count=30,
                                      min_n_iter=750, bandwidth=250, small_grad_th=1e-6):
        N = Y_T.shape[0]
        n_reg = X_T.shape[2]
        if n_beta_tv is not None:
            self.n_reg_beta_tv = n_beta_tv
        if dim_beta is not None:
            self.dim_beta = dim_beta
        N_beta_tv = self.n_reg_beta_tv * self.dim_beta
        n_reg_beta_c = n_reg - self.n_reg_beta_tv

        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val(dim_dist_par_un)

        if beta_const_0 is None:
            if not est_beta:
                raise

        if B0 is None:
            B0 = 0.95
            A0 = 0.01
            B, A = torch.tensor([B0, B0]), torch.ones(2) * A0
            wI, wO = 1 + torch.randn(N), torch.randn(N)
            w = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
            # add static SD pars for the time varying betas
            w = torch.cat((w, torch.zeros(N_beta_tv)))
            B = torch.cat((B, B0 * torch.ones(N_beta_tv)))
            A = torch.cat((A, A0 * torch.ones(N_beta_tv)))
            beta_const = torch.randn(dim_beta, n_reg-self.n_reg_beta_tv)*0.01
        else:
            B = B0
            A = A0
            w = W0
            beta_const = beta_const_0

        if init_filt_um:
            sd_par_0=None
        else:
            # use provided initialization or initialize at cross sectional estimate on the first obervation
            if sd_par_0 is None:
                X_0 = None
                if X_T is not None:
                    X_0 = X_T[:, :, :, 0]
                sd_par_0 = self.splitBA_init(w, B, A, init=True, Y_0=Y_T[:, :, 0], X_0=X_0)


        n = w.shape[0]
        N_BA = A.shape[0]

        if est_dis_par_un:  #if required estimate also the dist par
            n_dist_par_est = dim_dist_par_un
            if est_beta:
                unPar_0 = torch.cat((torch.cat((torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))), dist_par_un)),
                                     beta_const.view(-1)))#.clone().detach()

                def obj_fun(unPar):
                    reBA = self.un2re_BA_par(unPar[n:n + 2*N_BA])
                    return - self.loglike_sd_filt(unPar[:n], reBA[:N_BA], reBA[N_BA:], Y_T,
                                                dist_par_un=unPar[n + 2*N_BA:n + 2*N_BA + n_dist_par_est],
                                                beta_const=unPar[n + 2*N_BA + n_dist_par_est:].view(dim_beta, n_reg_beta_c),
                                                X_T=X_T)
            else:
                unPar_0 = torch.cat((torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))), dist_par_un))#.clone().detach()

                def obj_fun(unPar):
                    reBA = self.un2re_BA_par(unPar[n:n + 2*N_BA])
                    return - self.loglike_sd_filt(unPar[:n], reBA[:N_BA], reBA[N_BA:], Y_T,
                                                dist_par_un=unPar[n + 2*N_BA:n + 2*N_BA + n_dist_par_est],
                                                beta_const=beta_const.view(dim_beta, n_reg_beta_c),
                                                X_T=X_T)

        else:
            n_dist_par_est = 0
            unPar_0 = torch.cat((torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))),
                                 beta_const.view(-1)))#.clone().detach()

            def obj_fun(unPar):
                reBA = self.un2re_BA_par(unPar[n:n + 2 * N_BA])
                return - self.loglike_sd_filt(unPar[:n], reBA[:N_BA], reBA[N_BA:], Y_T,
                                            dist_par_un=dist_par_un,
                                            beta_const=unPar[n + 2 * N_BA:].view(dim_beta, n_reg_beta_c),
                                            X_T=X_T)

        unPar_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                      plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                      rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                      min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)

        w_est = unPar_est[:n].clone()
        re_BA_est = self.un2re_BA_par(unPar_est[n:n + 2*N_BA])
        if est_dis_par_un:
            dist_par_un_est = unPar_est[n + 2*N_BA:n + 2*N_BA + n_dist_par_est].clone()
        else:
            dist_par_un_est = dist_par_un
            n_dist_par_est=0
        if est_beta:
            beta_const_est = unPar_est[n + 2*N_BA + n_dist_par_est:].view(dim_beta, n_reg_beta_c)
        else:
            beta_const_est = beta_const_0
        return w_est.clone().detach(), re_BA_est[:N_BA].clone().detach(), \
               re_BA_est[N_BA:].clone().detach(), dist_par_un_est, \
               beta_const_est.clone().detach(), sd_par_0.clone().detach(), diag



def estimate_and_save_dirSpW1_models(Y_T, distribution, dim_dist_par_un, filter_type, regr_flag, SAVE_FOLD,
                                     X_T=None, dim_beta=None, n_beta_tv=0, unit_measure=1e6,
                                    learn_rate=0.01, T_test=10,
                                    N_steps=15000, print_every=1000, ovflw_lm=True, rescale_score=False,
                                     load_ss_beta_coeff=True, load_ss_as_0=False):

        Y_T = Y_T/unit_measure
        N = Y_T.shape[0]
        T = Y_T.shape[2]

        model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, distribution=distribution, rescale_SD=rescale_score)
        rel_improv_tol_SS = 1e-6
        min_n_iter_SS = 20
        bandwidth_SS = min_n_iter_SS
        no_improv_max_count_SS = 20
        rel_improv_tol_SD = 1e-7
        min_n_iter_SD = 750
        bandwidth_SD = 250
        no_improv_max_count_SD = 50
        if filter_type == 'SS':
            N_steps_iter = 50
            if not regr_flag:
                phi_ss_est_T, dist_par_un, beta, diag = \
                    model.ss_filt_est_beta_dist_par_const(Y_T, beta=None,
                                                          phi_T=None,
                                                          dist_par_un=None,
                                                          est_const_dist_par=True,
                                                          dim_dist_par_un=dim_dist_par_un,
                                                          est_const_beta=False,
                                                          dim_beta=1,
                                                          opt_large_steps=N_steps // N_steps_iter,
                                                          opt_steps_phi=N_steps_iter,
                                                          lRate_phi=learn_rate,
                                                          opt_steps_dist_par=N_steps_iter,
                                                          lRate_dist_par=learn_rate,
                                                          print_flag_phi=False,
                                                          print_flag_dist_par=False,
                                                          rel_improv_tol=rel_improv_tol_SS,
                                                          no_improv_max_count=no_improv_max_count_SS,
                                                          min_n_iter=min_n_iter_SS, bandwidth=bandwidth_SS,
                                                          small_grad_th=1e-6)

                if not(type(SAVE_FOLD) == list):
                    SAVE_FOLD_list = [SAVE_FOLD]
                else:
                    SAVE_FOLD_list = SAVE_FOLD

                for SAVE_FOLD in SAVE_FOLD_list:
                    try:
                        os.mkdir(SAVE_FOLD)
                    except:
                        pass

                    file_path = SAVE_FOLD + '/dirSpW1_SS_est_lr_' + \
                                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                                '_N_steps_' + str(N_steps) + \
                                '_ovflw_lm_' + str(ovflw_lm) + \
                                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                                distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'
                    print(file_path)

                    np.savez(file_path, phi_ss_est_T.detach(), dist_par_un.detach(), diag, N_steps, N_steps_iter,
                             unit_measure, learn_rate)

            else:
                if X_T is None:
                    raise
                #load starting values from single snapshot without regressors

                if load_ss_as_0:
                    learn_rate_load = 0.005
                    N_steps_load = 20000
                    file_path = SAVE_FOLD + '/dirSpW1_SS_est_lr_' + \
                                str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T) + \
                                '_N_steps_' + str(N_steps_load) + \
                                '_ovflw_lm_' + str(ovflw_lm) + \
                                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                                distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

                    ld_est = np.load(file_path, allow_pickle=True)
                    phi_T_0, dist_par_un_0 = tens(ld_est["arr_0"]), tens(ld_est["arr_1"])
                else:
                    phi_T_0 = model.start_phi_from_obs_T(Y_T)
                    dist_par_un_0 = model.dist_par_un_start_val(dim_dist_par_un)

                phi_ss_est_T, dist_par_un, beta_est, diag = \
                    model.ss_filt_est_beta_dist_par_const(Y_T, X_T=X_T, beta=None, phi_T=phi_T_0,
                                                          dist_par_un=dist_par_un_0,
                                                          est_const_dist_par=True, dim_dist_par_un=dim_dist_par_un,
                                                          est_const_beta=True, dim_beta=dim_beta,
                                                          opt_large_steps=N_steps//N_steps_iter,
                                                          opt_steps_phi=N_steps_iter, lRate_phi=learn_rate,
                                                          opt_steps_dist_par=N_steps_iter, lRate_dist_par=learn_rate,
                                                          opt_steps_beta=N_steps_iter, lRate_beta=learn_rate,
                                                          print_flag_phi=False, print_flag_dist_par=True,
                                                          print_flag_beta=True,
                                                          print_every=print_every,
                                                          rel_improv_tol=rel_improv_tol_SS,
                                                          no_improv_max_count=no_improv_max_count_SS,
                                                          min_n_iter=min_n_iter_SS, bandwidth=bandwidth_SS,
                                                          small_grad_th=1e-6)

                file_path = SAVE_FOLD + '/dirSpW1_X0_SS_est_lr_' + \
                            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                            '_N_steps_' + str(N_steps) + \
                            '_ovflw_lm_' + str(ovflw_lm) + \
                            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                            '_dim_beta_' + str(dim_beta) + \
                            distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'
                print(file_path)
                np.savez(file_path, phi_ss_est_T.detach(), dist_par_un.detach(), beta_est.detach(),
                         diag, N_steps, N_steps_iter,
                         unit_measure, learn_rate)

        if filter_type == 'SD':
            # if score driven estimate only on the observations that are not in the test sample
            Y_T = Y_T[:, :, :-T_test]
            if X_T is not None:
                X_T = X_T[:, :, :, :-T_test]
            N = Y_T.shape[0]
            T = Y_T.shape[2]
            N_BA = N
            learn_rate_load = 0.005
            N_steps_load = 20000

            if not regr_flag:
                # %load single snapshot estimates to be used as first values in the SD dynamics
                file_path = SAVE_FOLD + '/dirSpW1_SS_est_lr_' + \
                            str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T+T_test) + \
                            '_N_steps_' + str(N_steps_load) + \
                            '_ovflw_lm_' + str(ovflw_lm) + \
                            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                            distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

                ld_est = np.load(file_path, allow_pickle=True)
                phi_ss_est_T, dist_par_un_ss = tens(ld_est["arr_0"]), tens(ld_est["arr_1"])

                model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)
                # for t in range(T):
                #     phi_ss_est_T[:, t] = model.identify(phi_ss_est_T[:, t])
                phi_T_0 = model.start_phi_from_obs_T(Y_T[:, :, -10:])
                mean_ss_phi = phi_T_0.mean(dim=1)# phi_ss_est_T[:, 0]  # phi_ss_est_T[:, :T_test].mean(dim=1)

                # define initial points of optimizer: sets also the number of A,B parameters
                B0 = torch.ones(2 * N_BA) * 0.99
                A0 = torch.ones(2 * N_BA) * 0.00001
                W0 = mean_ss_phi * (1 - B0)
                dist_par_un_0 = dist_par_un_ss

                W_est, B_est, A_est, dist_par_un_est, sd_par_0, diag = model.estimate_SD(Y_T, B0=B0, A0=A0, W0=W0,
                                                                            opt_steps=N_steps,
                                                                            lRate=learn_rate,
                                                                            dist_par_un=dist_par_un_0,
                                                                            sd_par_0 = phi_ss_est_T[:, :5].mean(dim=1),
                                                                            dim_dist_par_un=dim_dist_par_un,
                                                                            est_dis_par_un=True,
                                                                            print_flag=True, print_every=print_every,
                                                                         rel_improv_tol=rel_improv_tol_SD,
                                                                         no_improv_max_count=no_improv_max_count_SD,
                                                                         min_n_iter=min_n_iter_SD,
                                                                         bandwidth=bandwidth_SD,
                                                                         small_grad_th=1e-6)


                if not(type(SAVE_FOLD) == list):
                    SAVE_FOLD_list = [SAVE_FOLD]
                else:
                    SAVE_FOLD_list = SAVE_FOLD

                for SAVE_FOLD in SAVE_FOLD_list:
                    try:
                        os.mkdir(SAVE_FOLD)
                    except:
                        pass

                    file_path = SAVE_FOLD + '/dirSpW1_SD_est_lr_' + \
                                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                                '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                                '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                                distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                                'test_sam_last_' + str(T_test) + '.npz'
                    print(file_path)
                    np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), dist_par_un_est.detach(),
                             sd_par_0.detach(), diag, N_steps, unit_measure, learn_rate)

            else:
                if X_T is None:
                    raise
                if load_ss_beta_coeff:
                    n_beta_tv=0
                    n_reg = X_T.shape[2]
                    file_path = SAVE_FOLD + '/dirSpW1_X0_SS_est_lr_' + \
                                str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T+T_test) + \
                                '_N_steps_' + str(N_steps_load) + \
                                '_ovflw_lm_' + str(ovflw_lm) + \
                                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                                '_dim_beta_' + str(dim_beta) + \
                                distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

                    ld_est = np.load(file_path, allow_pickle=True)
                    phi_ss_est_T, dist_par_un_0, beta_0 = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(
                        ld_est["arr_2"])

                    model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)
                    # for t in range(T):
                    #     phi_ss_est_T[:, t] = model.identify(phi_ss_est_T[:, t])
                    phi_T_0 = model.start_phi_from_obs_T(Y_T[:, :, -10:])
                    mean_ss_phi = phi_T_0.mean(dim=1)  # phi_ss_est_T[:, 0]  # phi_ss_est_T[:, :T_test].mean(dim=1)
                    est_beta=False
                    B0 = torch.ones(2*N_BA) * 0.99
                    A0 = torch.ones(2*N_BA) * 0.00001
                    W0 = mean_ss_phi * (1 - B0)
                    sd_par_0 = phi_ss_est_T[:, :5].mean(dim=1)
                else:
                    file_path = SAVE_FOLD + '/dirSpW1_X0_SD_est_lr_' + \
                                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                                '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                                '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                                distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                                '_dim_beta_' + str(dim_beta) + \
                                'test_sam_last_' + str(T_test) + '.npz'
                    ld_est = np.load(file_path, allow_pickle=True)
                    # define initial points of optimizer: sets also the number of A,B parameters
                    W0, B0, A0, dist_par_un_0,\
                    beta_0, sd_par_0 = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"]),\
                                       tens(ld_est["arr_3"]), tens(ld_est["arr_4"]), tens(ld_est["arr_5"])
                    est_beta = True



                W_est, B_est, A_est, dist_par_un_est, beta_est, sd_par_0, diag = model.estimate_SD_X0(Y_T, X_T, B0=B0,
                                                                                A0=A0, W0=W0,
                                                                                opt_steps=N_steps,
                                                                                lRate=learn_rate,
                                                                                est_dis_par_un=True,
                                                                                dist_par_un=dist_par_un_0,
                                                                                sd_par_0=sd_par_0,
                                                                                beta_const_0=beta_0,
                                                                                est_beta = est_beta,
                                                                                dim_beta=dim_beta,
                                                                                n_beta_tv=n_beta_tv,
                                                                                dim_dist_par_un=dim_dist_par_un,
                                                                                print_flag=True,
                                                                                plot_flag=False,
                                                                                print_every=print_every,
                                                                          rel_improv_tol=rel_improv_tol_SD,
                                                                          no_improv_max_count=no_improv_max_count_SD,
                                                                          min_n_iter=min_n_iter_SD,
                                                                          bandwidth=bandwidth_SD,
                                                                          small_grad_th=1e-6)

                file_path = SAVE_FOLD + '/dirSpW1_X0_SD_est_lr_' + \
                            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                            '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                            '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                            distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                            '_dim_beta_' + str(dim_beta) + \
                            'test_sam_last_' + str(T_test) + '.npz'
                print(file_path)
                np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), dist_par_un_est.detach(),
                         beta_est.detach(), sd_par_0.detach(), diag, N_steps, unit_measure, learn_rate)


def load_dirSpW1_models(N, T, distribution, dim_dist_par_un, filter_type, regr_flag, SAVE_FOLD,
                        dim_beta=None, n_beta_tv=0, unit_measure=1e6,
                        learn_rate=0.01, T_test=10,
                        N_steps=15000, ovflw_lm=True, rescale_score=False,
                        return_last_diag=False):

    if filter_type == 'SS':
        N_steps_iter = 100
        if not regr_flag:

            file_path = SAVE_FOLD + '/dirSpW1_SS_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '_ovflw_lm_' + str(ovflw_lm) + \
                        '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                        distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            phi_ss_est_T, dist_par_un, diag, N_steps, N_steps_iter,\
                                            unit_measure, learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                                                       l_dat["arr_2"], l_dat["arr_3"], \
                                                                       l_dat["arr_4"], l_dat["arr_5"], l_dat["arr_6"]

            beta_est = None
        else:
            file_path = SAVE_FOLD + '/dirSpW1_X0_SS_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '_ovflw_lm_' + str(ovflw_lm) + \
                        '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                        '_dim_beta_' + str(dim_beta) + \
                        distribution + '_distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            phi_ss_est_T, dist_par_un, beta_est,\
                     diag, N_steps, N_steps_iter,\
                        unit_measure, learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                                   l_dat["arr_2"], l_dat["arr_3"], \
                                                   l_dat["arr_4"], l_dat["arr_5"], \
                                                   l_dat["arr_6"], l_dat["arr_7"]

        if not return_last_diag:
            return  phi_ss_est_T, dist_par_un, beta_est,\
                    diag, N_steps, N_steps_iter,\
                    unit_measure, learn_rate
    if filter_type == 'SD':
        T=T-T_test
        N_BA = N
        if not regr_flag:
            file_path = SAVE_FOLD + '/dirSpW1_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                        '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                        '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                        distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                        'test_sam_last_' + str(T_test) + '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            W_est, B_est, A_est, dist_par_un_est, sd_par_0,\
                     diag, N_steps,\
                    unit_measure, learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                               l_dat["arr_2"], l_dat["arr_3"], \
                                               l_dat["arr_4"], l_dat["arr_5"], \
                                               l_dat["arr_6"], l_dat["arr_7"], \
                                               l_dat["arr_8"]
            beta_est = None

        else:

            file_path = SAVE_FOLD + '/dirSpW1_X0_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                        '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                        '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                        distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                        '_dim_beta_' + str(dim_beta) + \
                        'test_sam_last_' + str(T_test) + '.npz'
            print(file_path)

            l_dat = np.load(file_path)
            W_est, B_est, A_est, dist_par_un_est,\
                    beta_est, sd_par_0, diag, N_steps,\
                    unit_measure, learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                               l_dat["arr_2"], l_dat["arr_3"], \
                                               l_dat["arr_4"], l_dat["arr_5"], \
                                               l_dat["arr_6"], l_dat["arr_7"], \
                                               l_dat["arr_8"], l_dat["arr_9"]
        if not return_last_diag:
            return W_est, B_est, A_est, dist_par_un_est,\
                    beta_est, sd_par_0, diag, N_steps,\
                    unit_measure, learn_rate
    if return_last_diag:
        return diag[-1]











