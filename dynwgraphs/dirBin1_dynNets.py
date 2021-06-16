"""

dirBin1_Nets :  directed binary model with one parameter per node, aka theta model, directed configuration model, fitness model etc

                With the possibility to add one or more  regressors per link

"""

import torch
import sys
from .utils import splitVec, tens, putZeroDiag, optim_torch, gen_test_net, soft_lu_bound, soft_l_bound,\
    degIO_from_mat, strIO_from_mat, tic, toc, rand_steps, dgpAR, putZeroDiag_T
from torch.autograd import grad
import numpy as np
from .dirSpW1_dynNets import dirSpW1_dynNet_SD, dirSpW1_staNet
#-----------------------------------  Static model functions


class dirBin1_staNet(dirSpW1_staNet):
    def __init__(self, ovflw_lm=False, distribution='bernoulli'):
        dirSpW1_staNet.__init__(self, ovflw_lm=ovflw_lm, distribution=distribution)
        self.n_reg_beta_tv = 0
        self.dim_beta = 1
        self.ovflw_exp_L_limit = -50
        self.ovflw_exp_U_limit = 40

    # remove methods that do not make sense in the binary case
    def cond_exp_Y(self):
        """cond exp not needed for binary networks"""
        pass

    def link_dist_par(self, dist_par_un, N, A_t=None):
        return None

    def dist_par_un_start_val(self, dim_dist_par_un=1):
        return None

    def estimate_dist_par_const_given_phi_T(self, Y_T, phi_T, dim_dist_par_un, X_T=None, beta=None, dist_par_un_0=None, like_type=2, min_n_iter=200, opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False, print_every=10):
        return None, 0
   
    def invPiMat(self, phi, beta=None, X_t=None, ret_log=False):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(phi_i + phi_j)
        """
        phi_i, phi_o = splitVec(phi)
        log_inv_pi_mat = phi_i + phi_o.unsqueeze(1)
        if X_t is not None:
            log_inv_pi_mat = log_inv_pi_mat + self.regr_product(beta, X_t)
        if self.ovflw_lm:
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
                degIO = degIO_from_mat(Y_t)
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
                degIO = degIO_from_mat(Y_t)
            if (sum(degIO) == 0) | (sum(phi) == 0):
                raise
            par_i, par_o = splitVec(phi)
            zero_deg_par_i = - par_o.max() + torch.log(max_prob/N)
            zero_deg_par_o = - par_i.max() + torch.log(max_prob/N)

        return zero_deg_par_i, zero_deg_par_o

    def set_zero_deg_par(self, Y_t, phi_in, method="AVGSPACING", degIO=None):
        phi = phi_in.clone()
        if degIO is None:
            degIO = degIO_from_mat(Y_t)
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
            degIO = tens(degIO_from_mat(Y))
        ldeg = degIO.log()
        nnzInds = degIO > 0
        phi_0 = torch.ones(degIO.shape[0]) * (-15)
        phi_0[nnzInds] = degIO[nnzInds].log()
        for i in range(n_iter):
            phi_0 = self.phiFunc(phi_0, ldeg)
            #phi_0[~nnzInds] = -15
        phi_0 = self.set_zero_deg_par(Y, phi_0, degIO=degIO)
        return phi_0.clone()

    def dist_from_pars(self, distribution, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution matrix valued, from the model's parameters
        """
        p_mat = self.exp_A(phi, beta=beta, X_t=X_t)
        dist = torch.distributions.bernoulli.Bernoulli(p_mat)
        return dist

    def loglike_t(self, Y_t, phi, beta=None, X_t=None, dist_par_un=None, like_type=2, degIO=None):
        """
        The log likelihood of the zero beta model with or without regressors
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t).clone()
        if (like_type == 0) and (beta is None):
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

    def estimate_ss_t_bin_from_degIO(self, degIO, min_n_iter=200,
                                    opt_steps=5000, opt_n=1, lRate=0.01,
                                    print_flag=True, plot_flag=False, print_every=10):
        """
        single snapshot Maximum logLikelihood estimate of phi_t for binary networks using only the degrees - does no work with regressors
        """
        def obj_fun(unPar):
            return - self.loglike_t(torch.zeros(3, 3), unPar, degIO=degIO, like_type=0)
        nnzInds = degIO > 0

        unPar_0 = self.start_phi_from_obs(None, degIO=degIO)

        all_par_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                        min_n_iter=min_n_iter,
                                        plot_flag=plot_flag, print_flag=print_flag, print_every=print_every)

        #Identify the phi_t part
        all_par_est = self.identify(all_par_est)
        return all_par_est, diag


    def estimate_beta_const_given_phi_T(self, Y_T, X_T, phi_T, dim_beta, dist_par_un,  beta_0=None, like_type=0, opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False, print_every=10, rel_improv_tol=5 * 1e-7, no_improv_max_count=30, min_n_iter=150, bandwidth=50, small_grad_th=1e-6):
            """single snapshot Maximum logLikelihood estimate given beta_t"""
            T = Y_T.shape[2]
            n_reg = X_T.shape[2]
            if beta_0 is None:
                n_reg = X_T.shape[2]
                beta_0 = torch.zeros(dim_beta, n_reg)

            unPar_0 = beta_0.clone().detach()
            torch.autograd.set_detect_anomaly(True)
            # the likelihood for beta_t is the sum of all the single snapshots likelihoods, given phi_T
            def obj_fun(unPar):
                logl_T = 0
                for t in range(T):
                    logl_T = logl_T + self.loglike_t(Y_T[:, :, t], phi_T[:, t], X_t=X_T[:, :, :, t],
                                                       beta=unPar.view(dim_beta, n_reg),
                                                    like_type=like_type)
                return - logl_T.sum()

            beta_t_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                       plot_flag=plot_flag, print_flag=print_flag, print_every=print_every,
                                        rel_improv_tol=rel_improv_tol, no_improv_max_count=no_improv_max_count,
                                        min_n_iter=min_n_iter, bandwidth=bandwidth, small_grad_th=small_grad_th)


            return beta_t_est, diag

    def ss_filt_est_beta_const(self, Y_T, X_T=None, beta=None, phi_T=None, like_type=2,
                                        est_const_beta=False, dim_beta=1,
                                        opt_large_steps=10, opt_n=1, opt_steps_phi=500, lRate_phi=0.01,
                                        opt_steps_beta=400, lRate_beta=0.01,
                                        print_flag_phi=False, print_flag_beta=False,
                                       rel_improv_tol=5 * 1e-7, no_improv_max_count=30,
                                       min_n_iter=150, bandwidth=50, small_grad_th=1e-6):
        """
        use ss filt for za weighted version class without estimating dist_par
        """
        out = self.ss_filt_est_beta_dist_par_const(Y_T, X_T=X_T, beta=beta, phi_T=phi_T, dist_par_un=torch.ones(1),
                                                   like_type=like_type,
                                                   est_const_dist_par=False, dim_dist_par_un=torch.ones(1),
                                                   est_const_beta=est_const_beta, dim_beta=dim_beta,
                                                   opt_large_steps=opt_large_steps, opt_n=opt_n,
                                                   opt_steps_phi=opt_steps_phi, lRate_phi=lRate_phi,
                                                   opt_steps_dist_par=0, lRate_dist_par=0,
                                                   opt_steps_beta=opt_steps_beta, lRate_beta=lRate_beta,
                                                   print_flag_phi=print_flag_phi, print_flag_dist_par=False,
                                                   print_flag_beta=print_flag_beta,
                                                   min_n_iter=min_n_iter,
                                                   rel_improv_tol=rel_improv_tol,
                                                   no_improv_max_count=no_improv_max_count,
                                                    bandwidth=bandwidth,
                                                   small_grad_th=small_grad_th)
        return out



    def check_tot_exp(self, Y_t, phi_t, X_t=None, beta_t=None, one_dim_out=True):
        degIO = degIO_from_mat(Y_t)
        nnzInds = degIO != 0
        expMat = self.exp_A(phi_t, beta=beta_t, X_t=X_t)
        errsIO = (strIO_from_mat(expMat) - degIO)[nnzInds]
        relErrsIO = torch.div(torch.abs(errsIO), degIO[nnzInds])
        out = torch.abs(relErrsIO)  # check if the constraint is satisfied for all degs
        if one_dim_out:
            out = out.max()
        return out

    def exp_seq(self, phi_T, X_T=None, beta_const=None):
        N2, T = phi_T.shape
        N = N2//2
        E_Y_T = torch.zeros((N, N, T))
        X_t = None
        for t in range(T):
            if X_T is not None:
                X_t = X_T[:, :, :, t]
            phi_t = phi_T[:, t]
            E_Y_T[:, :, t] = self.exp_A(phi_t, beta=beta_const, X_t=X_t)
        return E_Y_T

    def dgp_phi_var_size(self, N, degMin=10, degMax=40, exponent=1):
        # exponent determines the convexity of the distributions of degrees between
        # deg min and deg max. increasing exponent --> decreasing density
        # if we let deg max grow with N than generate a degree distribution
        tmp = tens((np.linspace(degMin, degMax, N))**exponent)
        tmp1 = tmp * (degMax / tmp[-1])
        deg_io = torch.cat((tmp1, tmp1))
        deg_io[deg_io < degMin] = degMin
        um_phi, diag = self.estimate_ss_t_bin_from_degIO(deg_io, min_n_iter=500,
                                                         opt_steps=5000, opt_n=1, lRate=0.01,
                                                         print_flag=False, plot_flag=False, print_every=100)

        expDegs = strIO_from_mat(self.exp_A(um_phi))
        return um_phi, expDegs

    def sample_from_dgps(self, N, T, N_sample, dgp_type, X_T=None, n_reg=2, n_reg_beta_tv=1, dim_beta=1,
                         um_phi=None, degb=None):
        if um_phi is None:
            #if the unconditional means are not Given then fix them
            um_phi, um_degs = self.dgp_phi_var_size(N, degMin=degb[0], degMax=degb[1])

        um_phi = um_phi.detach()
        if degb is None:
            degb = tens([10, N-10])
        torch.manual_seed(2)
        phi_T = torch.zeros(2*N, T)
        beta_T = torch.zeros(N, T)
        if n_reg>0:
            if X_T is None:
                X_T = torch.distributions.normal.Normal(0.0, 1.0).sample((N, N, n_reg, T))
            X_0 = X_T[:, :, :, 0]

        N = um_phi.shape[0]//2

        period = T

        # define the oscillations bounds and phases fr phi
        um_phi_i, um_phi_o = splitVec(um_phi)

        # solve for x :deglb =  (N-1) * 1/(1+exp( - x - um_phi_o)) and obtain one upper and one lower bound for all in
        # and  out parameters
        bound_i = - um_phi_o.mean() - torch.log((N-1) / degb - 1)
        bound_o = - um_phi_i.mean() - torch.log((N-1) / degb - 1)
        ampl_i = torch.ones(N) * tens(np.linspace(0.4, 1, N)) * tens(np.diff(bound_i.detach()))
        ampl_o = torch.ones(N) * tens(np.linspace(0.4, 1, N)) * tens(np.diff(bound_o.detach()))
        ampl = torch.cat((ampl_i, ampl_o)).unsqueeze(1)
        # center the dynamical parameters at the middle of the bounds
        um_phi[:N] = ampl_i / 2 + bound_i[0]
        um_phi[N:] = ampl_o / 2 + bound_o[0]

        if dgp_type == 'sin':
            phi_T = um_phi.unsqueeze(1) + torch.sin(6.28*(tens(range(T))/period + torch.randn((2*N, 1)))) * ampl

            if n_reg>0:
                beta_T = torch.ones(dim_beta*n_reg, T)
            if n_reg_beta_tv>0:
                ampl_beta = 0.3
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                beta_T[:n_tv_beta_par, :] = beta_T[:n_tv_beta_par, :] + \
                                            torch.sin(6.28*tens(range(T))/period + torch.randn((n_tv_beta_par, 1))) * \
                                            ampl_beta

        if dgp_type == 'step':
            Nsteps = 2
            for n in range(2*N):
                minPar = um_phi[n] - ampl[n] / 2
                maxPar = um_phi[n] + ampl[n] / 2
                # steps between min and max values
                phi_T[n, :] = rand_steps(minPar, maxPar, Nsteps, T)

            if n_reg > 0:
                beta_T = torch.ones(dim_beta * n_reg, T)
            if n_reg_beta_tv > 0:
                ampl_beta = 0.3
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                beta_T[:n_tv_beta_par, T//2:] = beta_T[:n_tv_beta_par, T//2:] + ampl_beta

        if dgp_type == 'ar1':
            sigma = 0.01
            B = 0.999  # rand([0.7,0.99])
            scalType = "uniform"

            minPar = um_phi[n] - ampl[n] / 2
            maxPar = um_phi[n] + ampl[n] / 2
            phi_T = dgpAR(um_phi, B, sigma, T, minMax=[minPar, maxPar], scaling = scalType)
            if n_reg > 0:
                beta_T = torch.ones(dim_beta * n_reg, T)
            if n_reg_beta_tv > 0:
                n_tv_beta_par = n_reg_beta_tv * dim_beta
                w = (1 - B) * beta_T[:n_tv_beta_par, 0]
                beta_T[:n_tv_beta_par, 1:] = torch.randn((n_tv_beta_par, T-1)) * ampl
                for t in range(1, T):
                    beta_T[:n_tv_beta_par, t] == w + B * beta_T[:n_tv_beta_par, t-1] + beta_T[:n_tv_beta_par, t]


        #smaple given phi_T and p_T and identify phi_T

        Y_T = torch.zeros(N, N, T, N_sample)
        for t in range(T):
            phi = self.identify(phi_T[:, t])
            phi_T[:, t] = phi
            beta = beta_T[:, t].view(dim_beta, n_reg)
            X_t = X_T[:, :, :, t]

            dist = self.dist_from_pars('bernoulli', phi, beta, X_t, None)

            Y_t_S = dist.sample((N_sample,)).permute(1, 2, 0)
            Y_t_S = putZeroDiag_T(Y_t_S)
            Y_T[:, :, t, :] = Y_t_S

        return Y_T, phi_T, X_T, beta_T



class dirBin1_dynNet_SD(dirBin1_staNet, dirSpW1_dynNet_SD):
    """
        This a class  for directed binary dynamical networks, modelled with a bernoulli distribution,
        one dynamical parameter per each node (hence the 1 in the name) each evolving with a Score Driven
        Dynamics, and one regressor for each link. the number of parameters associated with the regressors
        is flexible but for the moment we have a single parameter equal for all links
        """

    def __init__(self, ovflw_lm=False, distribution='bernoulli', rescale_SD=False):
        dirBin1_staNet.__init__(self, ovflw_lm=ovflw_lm, distribution=distribution)
        self.rescale_SD = rescale_SD
        self.n_reg_beta_tv = 0
        self.dim_beta = 1
        self.backprop_sd = False

    def score_t(self, Y_t, phi_t, beta_t=None, X_t=None, dist_par_un=torch.zeros(1), backprop_score=False,
                like_type=2):
        """
        given the observations and the ZA gamma parameters (i.e. the cond mean
        matrix and the dist_par_un par), return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """
        N = phi_t.shape[0]//2
        A_t_bool = Y_t > 0
        A_t = tens(A_t_bool)
        if dist_par_un is not None:
            dist_par_un = dist_par_un.clone().detach()

        if beta_t is not None:
            beta = beta_t.clone().detach()
            beta.requires_grad = True
        else:
            beta = beta_t
        if backprop_score:
            phi = phi_t.clone().detach() # is the detach hebeta_tre harmful for the final estimation of SD static pars?????????
            phi.requires_grad = True
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi, beta=beta, X_t=X_t, dist_par_un=dist_par_un, like_type=like_type)
            score_phi = grad(like_t, phi, create_graph=True)[0]

        else:
            phi = phi_t.clone().detach()
            exp_A = self.exp_A(phi, beta=beta, X_t=X_t)

            tmp = A_t - exp_A

            if self.rescale_SD:
                diag_resc_mat = exp_A * (1 - exp_A)

            if self.ovflw_lm:
                log_inv_pi__mat = self.invPiMat(phi, ret_log=True)
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

        score_i, score_o = splitVec(score_phi)
        score_beta = None
        if beta_t is not None:
            if self.n_reg_beta_tv > 0:
                score_beta = grad(like_t, beta, create_graph=True)[0][:, :self.n_reg_beta_tv].view(-1)

        return score_i, score_o, score_beta

    def sd_dgp(self, w, B, A, N=None, T=None, beta_const=None, sd_par_0=None, X_T=None, dist_par_un=None):
        """given the static parameters, sample the dgp with a score driven dynamics.
        equal to the filter except sample the observations for each t.
        """

        N_beta_tv = self.n_reg_beta_tv * self.dim_beta

        Y_T = torch.zeros((N, N, T))
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
            #-----------------------------------
            # The following lines must be the only diff in the loop between dgp and
            phi_t = sd_par_t[:2*N]
            if N_beta_tv > 0:
                beta_t = torch.cat((sd_par_t[-N_beta_tv:].view(self.dim_beta, self.n_reg_beta_tv), beta_const), dim=1)
            else:
                beta_t = beta_const
            dist = self.dist_from_pars(self.distr, phi_t, beta_t, X_t, dist_par_un)
            Y_t = dist.sample()
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




def estimate_and_save_dirBin1_models(Y_T, filter_type, regr_flag, SAVE_FOLD, X_T=None, dim_beta=None,
                                learn_rate=0.01, T_test=10,
                                N_steps=15000, print_every=1000, ovflw_lm=True, rescale_score=False):


    N = Y_T.shape[0]
    T = Y_T.shape[2]

    model = dirBin1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score)
    phi_T_0 = torch.zeros(N * 2, T)
    for t in range(T):
        phi_T_0[:, t] = model.start_phi_form_obs(Y_T[:, :, t])

    rel_improv_tol_SS = 1e-6
    min_n_iter_SS = 20
    bandwidth_SS = min_n_iter_SS
    no_improv_max_count_SS = 20
    rel_improv_tol_SD = 1e-7
    min_n_iter_SD = 750
    bandwidth_SD = 250
    no_improv_max_count_SD = 50

    learn_rate_load = 0.01
    N_steps_load = 20000

    if filter_type == 'SS':

        if not regr_flag:
            phi_ss_est_T, diag = model.ss_filt(Y_T, phi_T_0=phi_T_0,
                                               est_dist_par=False, est_beta=False,
                                               opt_steps=N_steps, lRate=learn_rate, print_flag=True,
                                               print_every=print_every,
                                               rel_improv_tol=rel_improv_tol_SS,
                                               no_improv_max_count=no_improv_max_count_SS,
                                               min_n_iter=min_n_iter_SS, bandwidth=bandwidth_SS,
                                               small_grad_th=1e-6)

            file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '.npz'
            print(file_path)
            np.savez(file_path, phi_ss_est_T.detach(), None, diag)

        else:
            if X_T is None:
                raise
            N_steps_iter = 200
            learn_rate_beta = learn_rate/2
            file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
                        str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T ) + \
                        '_N_steps_' + str(N_steps_load) + \
                        '.npz'
            ld_est = np.load(file_path, allow_pickle=True)
            phi_ss_est_T_0 = tens(ld_est["arr_0"])

            phi_ss_est_T, dist_par_un, beta_est, diag_joint = \
                model.ss_filt_est_beta_const(Y_T,
                                             X_T=X_T,
                                             beta=torch.zeros(dim_beta, 1),
                                             phi_T=phi_ss_est_T_0,
                                             est_const_beta=True,
                                             dim_beta=dim_beta,
                                             opt_large_steps=N_steps // N_steps_iter,
                                             opt_steps_phi=N_steps_iter,
                                             lRate_phi=learn_rate,
                                             opt_steps_beta=N_steps_iter,
                                             lRate_beta=learn_rate_beta,
                                             print_flag_phi=False,
                                             print_flag_beta=True,
                                             rel_improv_tol=rel_improv_tol_SS,
                                             no_improv_max_count=no_improv_max_count_SS,
                                             min_n_iter=min_n_iter_SS, bandwidth=bandwidth_SS,
                                             small_grad_th=1e-6)

            file_path = SAVE_FOLD + '/dirBin1_X0_SS_est_lr_phi' + \
                        str(learn_rate) + '_lr_beta_' + str(learn_rate_beta) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '_dim_beta_' + str(dim_beta) + '.npz'
            print(file_path)
            np.savez(file_path, phi_ss_est_T.detach(), beta_est.detach(), diag_joint,
                     N_steps, learn_rate, learn_rate_beta)

    if filter_type == 'SD':
        # if score driven estimate only on the observations that are not in the test sample
        Y_T = Y_T[:, :, :-T_test]
        if X_T is not None:
            X_T = X_T[:, :, :, :-T_test]
        N = Y_T.shape[0]
        T = Y_T.shape[2]
        N_BA = N
        file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
                    str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T + T_test) + \
                    '_N_steps_' + str(N_steps_load) + \
                    '.npz'
        ld_est = np.load(file_path, allow_pickle=True)
        phi_ss_est_T_0 = tens(ld_est["arr_0"])


        B0 = torch.ones(2*N_BA) * 0.98
        A0 = torch.ones(2*N_BA) * 0.01
        W0 = phi_ss_est_T_0.mean(dim=1) * (1 - B0)
        if not regr_flag:
            W_est, B_est, A_est, dist_par_un_est, sd_par_0, diag = model.estimate_SD(Y_T, B0=B0, A0=A0, W0=W0, opt_steps=N_steps, lRate=learn_rate, sd_par_0=phi_ss_est_T_0[:, :5].mean(dim=1), print_flag=True, print_every=print_every, plot_flag=False, est_dis_par_un=False, init_filt_um=False, rel_improv_tol=rel_improv_tol_SD, no_improv_max_count=no_improv_max_count_SD, min_n_iter=min_n_iter_SD, bandwidth=bandwidth_SD, small_grad_th=1e-6)

            file_path = SAVE_FOLD + '/dirBin1_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + '.npz'
            print(file_path)
            np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), None, sd_par_0, diag,
                     N_steps, learn_rate)

        else:
            if X_T is None:
                raise
            n_beta_tv=0
            n_reg = X_T.shape[2]
            W_est, B_est, A_est, dist_par_un_est, beta_const_est, sd_par_0, diag = \
                model.estimate_SD_X0(Y_T, X_T=X_T,
                                     dim_beta=dim_beta, n_beta_tv=n_beta_tv,
                                     B0=torch.cat((B0, torch.zeros(n_beta_tv))),
                                     A0=torch.cat((A0, torch.zeros(n_beta_tv))),
                                     W0=torch.cat((W0, torch.zeros(n_beta_tv))),
                                     beta_const_0=torch.randn(dim_beta, n_reg - n_beta_tv)*0.01,
                                     opt_steps=N_steps,
                                     lRate=learn_rate,
                                     sd_par_0=phi_ss_est_T_0[:, :5].mean(dim=1),
                                     print_flag=True,
                                     print_every=print_every,
                                     plot_flag=False,
                                     rel_improv_tol=rel_improv_tol_SD,
                                     no_improv_max_count=no_improv_max_count_SD,
                                     min_n_iter=min_n_iter_SD,
                                     bandwidth=bandwidth_SD,
                                     small_grad_th=1e-6)

            file_path = SAVE_FOLD + '/dirBin1_X0_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                        '_dim_beta_' + str(dim_beta) + '.npz'

            print(file_path)

            np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), beta_const_est.detach(), sd_par_0,
                     diag, N_steps, learn_rate)


def load_dirBin1_models(N, T, filter_type, regr_flag, SAVE_FOLD,
                        dim_beta=None, n_beta_tv=0, learn_rate_beta=0.005,
                        learn_rate=0.01, T_test=10,
                        N_steps=15000, ovflw_lm=True, rescale_score=False,
                        return_last_diag=False):

    if filter_type == 'SS':
        N_steps_iter = 100
        if not regr_flag:

            file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            phi_ss_est_T, diag = l_dat["arr_0"], l_dat["arr_1"]
            beta_est = None
        else:
            file_path = SAVE_FOLD + '/dirBin1_X0_SS_est_lr_phi' + \
                        str(learn_rate) + '_lr_beta_' + str(learn_rate_beta) + '_N_' + str(N) + '_T_' + str(T) + \
                        '_N_steps_' + str(N_steps) + \
                        '_dim_beta_' + str(dim_beta) + '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            phi_ss_est_T, beta_est,\
                     diag, N_steps,\
                     learn_rate, learn_rate_beta = l_dat["arr_0"], l_dat["arr_1"], \
                                                   l_dat["arr_2"], l_dat["arr_3"], \
                                                   l_dat["arr_4"], l_dat["arr_5"]

        if not return_last_diag:
            return  phi_ss_est_T, beta_est,\
                    diag, N_steps, N_steps_iter, learn_rate
    if filter_type == 'SD':
        T=T-T_test
        N_BA = N
        if not regr_flag:
            file_path = SAVE_FOLD + '/dirBin1_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + '.npz'
            print(file_path)
            l_dat = np.load(file_path)
            W_est, B_est, A_est, beta_est, sd_par_0, diag, N_steps,\
                     learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                               l_dat["arr_2"], l_dat["arr_3"], \
                                               l_dat["arr_4"], l_dat["arr_5"], \
                                               l_dat["arr_6"], l_dat["arr_7"]

            beta_est = None
        else:

            file_path = SAVE_FOLD + '/dirBin1_X0_SD_est_lr_' + \
                        str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
                        '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                        '_dim_beta_' + str(dim_beta) + '.npz'

            print(file_path)

            l_dat = np.load(file_path)
            W_est, B_est, A_est, \
                    beta_est, sd_par_0, diag, N_steps,\
                    learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
                                   l_dat["arr_2"], l_dat["arr_3"], \
                                   l_dat["arr_4"], l_dat["arr_5"], \
                                   l_dat["arr_6"], l_dat["arr_7"]

        if not return_last_diag:
            return W_est, B_est, A_est, \
                    beta_est, sd_par_0, diag, N_steps,\
                     learn_rate
    if return_last_diag:
        return diag[-1]

















