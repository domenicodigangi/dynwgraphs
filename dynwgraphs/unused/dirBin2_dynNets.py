#########
#Created Date: Monday March 15th 2021
#Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
#-----
#Last Modified: Sunday June 6th 2021 4:27:20 pm
#Modified By:  Domenico Di Gangi
#-----
#Description: Functions for Zero Augmented models for Score driven erdos reny, coded using pytorch
#-----
########


"""
TO DO:  1) DONE- add regressors
        2) consider diferent distributions

"""
import torch
import sys
import numpy as np
sys.path.append("./src/")
from .utils import splitVec, tens, putZeroDiag, putZeroDiag_T, optim_torch, gen_test_net, soft_lu_bound, soft_l_bound,\
    degIO_from_mat, strIO_from_mat, tic, toc
from torch.autograd import grad
#----------------------------------- Zero Augmented Static model functions
#self = dirBin2_dynNet_SD(ovflw_lm=True, rescale_SD = False )



class dirBin2_staNet(object):
    """
    This a class  for directed weighted sparse static networks (or sequences), modelled with a zero augmented
    distribution, one  parameter per each node (hence the 1 in the name)
    """

    def __init__(self, ovflw_lm=True, distribution='bernoulli', N=10):
        self.ovflw_lm = ovflw_lm
        self.distr = distribution
        self.N = N

    def invPiMat(self, phi, ret_log=False):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(phi_i + phi_j)
        """
        log_inv_pi_mat = phi
        if self.ovflw_lm:
            """if required force the exponent to stay whithin overflow-safe bounds"""
            log_inv_pi_mat = soft_lu_bound(log_inv_pi_mat)

        if ret_log:
            return log_inv_pi_mat
        else:
            return torch.exp(log_inv_pi_mat)


    def exp_A(self, phi):
        """
        given the vector of unrestricted parameters, compute the expected
        matrix, can have non zero elements on the diagonal!!
        """
        invPiMat_= self.invPiMat(phi)
        out = invPiMat_/(1 + invPiMat_)
        return out * torch.ones((self.N, self.N))

    def dist_from_pars(self, phi):
        """
        return a pytorch distribution matrix valued
        """
        p_mat = self.exp_A(phi)
        dist = torch.distributions.bernoulli.Bernoulli(p_mat)
        return dist


    def loglike_t(self, Y_t, phi):
        """
        The log likelihood of the zero beta model with or without regressors
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t).clone()
        # compute the likelihood using torch buit in functions
        dist = self.dist_from_pars( phi)
        log_probs = dist.log_prob(Y_t).clone()
        out = torch.sum(log_probs)
        return out.clone()



class dirBin2_dynNet_SD(dirBin2_staNet):
    """
        This a class  for directed weighted sparse dynamical networks, modelled with a zero augmented distribution,
        one dynamical parameter per each node (hence the 1 in the name) each evolving with a Score Driven
        Dynamics, and one regressor for each link. the number of parameters associated with the regressors
        will be flexible but for the moment we have a single parameter equal for all links
        """

    def __init__(self, ovflw_lm=False, distribution='bernoulli', rescale_SD=False, N=10):
        dirBin2_staNet.__init__(self, ovflw_lm=ovflw_lm, distribution=distribution)
        self.rescale_SD = rescale_SD
        self.N = N


    @staticmethod
    def un2re_BA_par( BA_un):
        B_un, A_un = splitVec(BA_un)
        exp_B = torch.exp(B_un)
        return torch.cat((torch.div(exp_B, (1 + exp_B)), torch.exp(A_un)))

    @staticmethod
    def re2un_BA_par(BA_re):
        B_re, A_re = splitVec(BA_re)
        return torch.cat((torch.log(torch.div(B_re, 1 - B_re)), torch.log(A_re)))


    def score_t(self, Y_t, phi_t, backprop_score = False):
        """
        """
        rescale = self.rescale_SD
        # compute the score with AD using Autograd
        phi = phi_t.clone().detach()# is the detach hebeta_tre harmful for the final estimation of SD static pars?????????

        if backprop_score:
            phi.requires_grad = True
            like_t = self.loglike_t(Y_t, phi)
            score_phi = grad(like_t, phi, create_graph=True)[0]
            #rescale score if required
            if rescale:
                raise# missing rescaling routines for beta_t
                drv2 = []
                for i in range(phi.shape[0]):
                    tmp = score_phi[i]
                    # compute the second derivatives of the loglikelihood
                    drv2.append(grad(tmp, phi, retain_graph=True)[0][i])
                drv2 = torch.stack(drv2)
                # diagonal rescaling
                scaled_score = score_phi / (torch.sqrt(-drv2))
            else:
                scaled_score = score_phi
        else:
            exp_A = self.exp_A(phi).sum()
            score_phi = Y_t.sum() - exp_A
            if self.rescale_SD:
                diag_resc_mat = exp_A * (1 - exp_A)
                scaled_score = score_phi/diag_resc_mat.sum().sqrt()
            else:
                scaled_score = score_phi

        return scaled_score


    def update_dynw_par(self, Y_t, par_sd_t, w, B, A):
        """
        score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
        """
        phi_t = par_sd_t

        s = self.score_t(Y_t, phi_t)
        phi_tp1 = w + B*phi_t + A*s

        return phi_tp1.clone(), s

    def sd_dgp(self, w, B, A, N, T):

        Y_T = torch.zeros((N, N, T))
        sd_par_T = torch.zeros(1)

        # initialize the time varying parameters to the unconditional mean
        sd_par_t = w/(1-B)
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            sd_par_T = torch.cat((sd_par_T, sd_par_t), 0)

            #-----------------------------------
            # The following lines must be the only diff in the loop between dgp and
            phi_t = sd_par_t
            dist = self.dist_from_pars( phi_t)
            Y_t = dist.sample()
            Y_t = putZeroDiag(Y_t)
            Y_T[:, :, t] = Y_t.clone()
            #----------------------------------

            sd_par_tp1, score = self.update_dynw_par(Y_t, sd_par_t, w, B, A)
            sd_par_t = sd_par_tp1.clone()
            # print(torch.isnan(phi_tp1).sum())

        phi_T = sd_par_T[1:]

        return phi_T, Y_T

    def sd_filt(self, w, B, A, Y_T):
        T = Y_T.shape[2]
        sd_par_T = torch.zeros(1)
        # initialize the time varying parameters to the unconditional mean
        sd_par_t = w / (1 - B)
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            sd_par_T = torch.cat((sd_par_T, sd_par_t), 0)
            # -----------------------------------
            # The following lines must be the only diff in the loop between dgp and
            Y_t = Y_T[:, :, t]
            # ----------------------------------

            sd_par_tp1, score = self.update_dynw_par(Y_t, sd_par_t, w, B, A)
            sd_par_t = sd_par_tp1.clone()
            # print(torch.isnan(phi_tp1).sum())
        phi_T = sd_par_T[1:]
        return phi_T

    def loglike_sd_filt(self, w, B, A, Y_T):
        phi_T = self.sd_filt(w, B, A, Y_T)
        T = Y_T.shape[2]
        logl_T = 0
        for t in range(T):
            Y_t = Y_T[:, :, t]
            phi_t = phi_T[t]

            logl_T = logl_T + self.loglike_t(Y_t, phi_t)
        return logl_T


    def estimate_SD(self, Y_T, opt_n=1, max_opt_iter=800, lr=0.005, plot_flag=False, print_flag=False,
                    B0=None, A0=None, W0=None, print_every=200):

        if B0 is None:
            B, A = torch.tensor(0.7), torch.ones(1) * 0.0001
            w = 1 + torch.randn(1)
        else:
            B = B0
            A = A0
            w = W0

        unPar_0 = torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))).clone().detach()
        def obj_fun(unPar):
            reBA = self.un2re_BA_par(unPar[1:])
            return - self.loglike_sd_filt(unPar[:1], reBA[:1], reBA[1:], Y_T,)

        unPar_est, diag = optim_torch(obj_fun, unPar_0, max_opt_iter=max_opt_iter, opt_n=opt_n, lr=lr,
                                      plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                        rel_improv_tol = 1e-6, no_improv_max_count=20)

        w_est = unPar_est[:1].clone()
        re_BA_est = self.un2re_BA_par(unPar_est[1:])

        return w_est.clone(), re_BA_est[0].clone(), re_BA_est[1].clone(), diag












