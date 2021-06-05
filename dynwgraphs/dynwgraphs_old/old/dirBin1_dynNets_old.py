"""

dirBin1_Nets :  directed binary model with one parameter per node, aka delta
                model, directed configuration model, fitness model etc

dirBin1_X0_Nets :  add the possibility to add one regressor per link

"""
import torch
import sys
sys.path.append("./src/")
from utils import splitVec, tens, putZeroDiag, optim_torch, gen_test_net, soft_lu_bound, soft_l_bound,\
    degIO_from_mat, strIO_from_mat, tic, toc
import matplotlib.pyplot as plt
from torch.autograd import grad
#----------------------------------- Zero Augmented Static model functions


class dirBin1_staNet(object):
    """
    This a class  for binary directed networks (or sequences), modelled with bernoulli
    distribution, one  parameter per each node (hence the 1 in the name)
    """

    def __init__(self):
        pass

    @staticmethod
    def invPiMat(theta, delta=None, X_t=None):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(theta_i + theta_j)
        """
        theta_i, theta_o = splitVec(theta)
        return putZeroDiag(torch.mul(torch.exp(theta_i), torch.exp(theta_o).unsqueeze(1)))

    def expMatBin(self, theta, delta=None, X_t=None):
        """
        given the vector of unrestricted parameters, compute the expected
        matrix
        """
        invPiMat = self.invPiMat(theta, delta=delta, X_t=X_t)
        return torch.div(invPiMat, (1 + invPiMat))

    @staticmethod
    def zeroDegParFunBin(degIO, parIO=None):
        zeroDegPar = -1e6
        return zeroDegPar, zeroDegPar

    @staticmethod
    def phiFunc(vParUn, ldegI, ldegO):
        """ function that defines one iteration of the estimation map
        of Chatterjee, Diaconis and  Sly for the directed delta model
        """
        vParReI, vParReO = splitVec(torch.exp(vParUn))
        vParReO = vParReO.unsqueeze(1)
        matI = putZeroDiag(1 / ((1 / vParReI) + vParReO))
        matO = putZeroDiag(1 / (vParReI + (1 / vParReO)))
        outPhiI = (ldegI - torch.log(torch.sum(matI, axis=1)))
        outPhiO = (ldegO - torch.log(torch.sum(matO, 0)))
        return torch.cat((outPhiI, outPhiO))

    def loglike_bin_t(self, A_t, theta_t, fast_mle=False, degIO=None, delta=None, X_t=None):
        """
        log likelihood of the binary matrix
        """
        if (X_t is None) & fast_mle:
            degIO = strIO_from_mat(A_t)
            tmp1 = torch.sum(torch.mul(theta_t, degIO))
            tmp2 = torch.sum(torch.log(1 + self.invPiMat(theta_t, delta=delta, X_t=X_t)))
            llike_t = tmp1 - tmp2
        else:
            # for a bernoulli the matrix of probabilties is the expected matrix
            p_mat = self.expMatBin(theta_t, delta=delta, X_t=X_t)
            dist = torch.distributions.bernoulli.Bernoulli(probs=p_mat)
            log_probs = dist.log_prob(A_t)
            llike_t = torch.sum(log_probs[~torch.isnan(log_probs)])
        return llike_t

    def estimate_ss_bin_t_MLE(self, A_t, theta_0=None, fast_mle=False, X_t=None, delta=None, opt_steps=150,
                        lRate=0.5, opt_n=1, plot_flag=False, print_flag=False, print_fun=None):
        """Estimate model's parameters by optimizing the loglikelihood"""
        degIO = strIO_from_mat(A_t)
        if theta_0 is None:
            theta_0 = torch.log(degIO)
            theta_0[degIO==0] = -20
        unPar_0 = theta_0.clone().detach()
        if print_fun is None:
            print_fun = lambda x: torch.mean(self.check_fitBin(x, degIO))

        def obj_fun(unPar):
            return - self.loglike_bin_t(A_t, unPar, fast_mle=fast_mle, degIO=degIO, X_t=X_t, delta=delta)

        theta_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                      plot_flag=plot_flag, print_flag=print_flag, print_every=20,
                                      print_fun=print_fun)
        return theta_est, diag

    def check_fitBin(self, theta, degIO, delta=None, X_t=None, nnzInds=None):
        if nnzInds is None:
            nnzInds = degIO != 0
        expMat = self.expMatBin(theta, delta=delta, X_t=X_t)
        errsIO = (strIO_from_mat(expMat) - degIO)[nnzInds]
        relErrsIO = torch.div(torch.abs(errsIO), degIO[nnzInds])
        return torch.abs(relErrsIO)  # check if the constraint is satisfied for all degs

    def estimate_ss_bin_t(self, A_t, fast_mle=False, X_t=None,
                    targetErr=1e-2, plot_flag=False, print_flag=False,  maxIt=50, max_mle_cycle=30, mle_steps=50):
        """ estimate the directer delta model (aka fitness, aka configuration)
        following the very fast approach of Chatterjee, Diaconis and  Sly
        """
        degIO = strIO_from_mat(A_t)
        degI, degO = splitVec(degIO)
        N = degI.shape[0]
        ldegI = torch.log(degI)
        ldegO = torch.log(degO)
        # L = torch.sum(degI)
        # LperLink = L/(N**2 - N)
        # unifstartval =  0.5*torch.log((LperLink/((1-LperLink))))
        nnzInds = degIO != 0

        #    if (torch.sum(degI) - torch.sum(degO) ) > 0.1 :
        #        raise Exception("sums of in and out degrees should be equal")


        it = 0
        vParUn = torch.cat((ldegI, ldegO))
        err_list = []
        relErrMax = 10
        while (relErrMax > targetErr) & (it < maxIt) :
            vParUn = self.phiFunc(vParUn, ldegI, ldegO)
            relErrMax = torch.max(self.check_fitBin(vParUn, degIO, nnzInds=nnzInds))
            relErrMean = torch.mean(self.check_fitBin(vParUn, degIO, nnzInds=nnzInds))
            err_list.append((relErrMax.item(), relErrMean.item()))
            it += 1
            # print(it)


        zeroParI, zeroParO = self.zeroDegParFunBin(degIO)

        vParMle = vParUn.clone()
        vParMle[0:N][degIO[0:N] == 0] = zeroParI
        vParMle[N:2 * N][degIO[N:2 * N] == 0] = zeroParO
        mle_cycle = 0
        improv = 10
        if relErrMax > targetErr:
            while (relErrMax > targetErr) & (mle_cycle < max_mle_cycle)& (improv > 1e-3):
                vParMle, diag_mle = self.estimate_ss_bin_t_MLE(A_t, vParMle, fast_mle=fast_mle, opt_steps=mle_steps, opt_n=1, plot_flag=plot_flag)
                relErrMax_n = torch.max(self.check_fitBin(vParMle, degIO, nnzInds=nnzInds))
                relErrMean_n = torch.mean(self.check_fitBin(vParMle, degIO, nnzInds=nnzInds))
                improv = (relErrMax - relErrMax_n)
                relErrMax = relErrMax_n
                err_list.append((relErrMax_n.item(), relErrMean_n.item()))
                if print_flag:
                    print(relErrMax.item(), torch.mean(self.check_fitBin(vParMle, degIO, nnzInds)).item())
                mle_cycle += 1
        return vParMle, err_list

    @staticmethod
    def sampleBin(p_mat, delta=None, X_t=None):
        """
        given the matrix of probabilities of observing a link
        sample a binary matrix
        """
        return torch.bernoulli(p_mat) > 0

    @staticmethod
    def identify(theta, id_type=1):
        """
         enforce an identification condition on the parameters of the delta model
        """
        # set the first in parameter to zero
        theta_i, theta_o = splitVec(theta)
        if id_type == 0:
            """set one to zero"""
            d = theta_i[0]
        elif id_type == 1:
            """set in and out sum to zero """
            d = (theta_i.mean() - theta_o.mean()) / 2

        theta_i_out = theta_i - d
        theta_o_out = theta_o + d
        return torch.cat((theta_i_out, theta_o_out))

    def ss_filt_bin(self, A_T, X_T=None, targetErr=1e-2, lRate=0.5, opt_steps=50, print_flag=False, mle_only=False):
        """
        Static sequence filter.
        return the sequence  of single snapshot estimates
        """
        T = A_T.shape[2]
        N = A_T.shape[0]
        diag_T=[]
        theta_T = torch.zeros((2 * N, T))
        X_t = None
        for t in range(T):
            print(t)
            A_t = A_T[:, :, t]
            if X_T is not None:
                X_t = X_T[:, :, t]
            if mle_only:
                theta_t, diag = self.estimate_ss_bin_t_MLE(A_t, X_t=X_t, print_flag=print_flag,
                                                       opt_steps=opt_steps, lRate=lRate)
                tmp = self.check_fitBin(theta_t, strIO_from_mat(A_t))
                diag= (tmp.max().item(), tmp.mean().item())
            else:
                theta_t, diag = self.estimate_ss_bin_t(A_t, X_t=X_t, targetErr=targetErr, print_flag=print_flag,
                                                               max_mle_cycle=opt_steps)
            diag_T.append(diag)
            theta_t = self.identify(theta_t)
            theta_T[:, t] = theta_t
        return theta_T, diag_T

    def like_seq(self, A_T, theta_T, X_T=None, delta=None):
        T = A_T.shape[2]
        like_seq = 0
        X_t = None
        for t in range(T):
            # estimate each theta_t given delta
            A_t = A_T[:, :, t]
            if X_T is not None:
                X_t = X_T[:, :, t]
            theta_t = theta_T[:, t]
            like_seq = like_seq + self.loglike_bin_t(A_t, theta_t, delta=delta, X_t=X_t)
        return like_seq


class dirBin1_dynNet_SD(dirBin1_staNet):
    """
       This a class  for binary directed  dynamical networks, modelled with a Score driven delta model, i.e.
       one  parameter per each node (hence the 1 in the name), each evolving with a score driven dynamics
       """
    def __init__(self):
        dirBin1_staNet.__init__(self)

    @staticmethod
    def un2re_BA_par(BA_un):
        B_un, A_un = splitVec(BA_un)
        exp_B = torch.exp(B_un)
        return torch.cat((torch.div(exp_B, (1 + exp_B)), torch.exp(A_un)))

    @staticmethod
    def re2un_BA_par(BA_re):
        B_re, A_re = splitVec(BA_re)
        return torch.cat((torch.log(torch.div(B_re, 1 - B_re)), torch.log(A_re)))

    def score_bin_t_AD(self, A_t, theta_t, delta=None, X_t=None, rescale=False):
        """
        given the observations return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """
        # compute the score with AD using Autograd
        theta = theta_t.clone().detach()# is the detach here harmful for the final estimation of SD static pars?????????
        theta.requires_grad = True
        like_t = self.loglike_bin_t(A_t, theta, delta=delta, X_t=X_t)
        score = grad(like_t, theta, create_graph=True)[0]
        # rescale score if required
        if rescale:
            drv2 = []
            for i in range(theta.shape[0]):
                tmp = score[i]
                # compute the second derivatives of the loglikelihood
                drv2.append(grad(tmp, theta, retain_graph=True)[0][i])
            drv2 = torch.stack(drv2)
            # diagonal rescaling
            scaled_score = score / (torch.sqrt(-drv2))
        else:
            scaled_score = score
        score_i, score_o = splitVec(scaled_score)
        return score_i, score_o

    def update_dynbin_par(self, A_t, theta_t, w, B, A, delta=None, X_t=None):
        """
        score driven update of the node specific parameters
        Identify the vector before updating and after
        """
        N = A_t.shape[0]
        theta_t = self.identify(theta_t)
        theta_i_t, theta_o_t = theta_t[:N], theta_t[N:2 * N]
        w_i, w_o = splitVec(w)
        B_i, B_o = splitVec(B)
        A_i, A_o = splitVec(A)
        s_i, s_o = self.score_bin_t_AD(A_t, theta_t, delta=delta, X_t=X_t)

        theta_i_tp1 = w_i + torch.mul(B_i, theta_i_t) + torch.mul(A_i, s_i)
        theta_o_tp1 = w_o + torch.mul(B_o, theta_o_t) + torch.mul(A_o, s_o)
        theta_tp1 = torch.cat((theta_i_tp1, theta_o_tp1))

        return self.identify(theta_tp1), torch.cat((s_i, s_o))

    def sd_dgp_bin(self, w, B, A, T, delta=None, X_T=None):
        """
        given the static parameters, sample the dgp with a score driven dynamics.
        equal to the filter except sample the observations for each t.
        """
        w_i, w_o = splitVec(w)
        B_i, B_o = splitVec(B)

        N = w_i.shape[0]

        A_T = torch.zeros((N, N, T))
        theta_T = torch.zeros((2 * N, 1))
        # initialize the time varying parameters to the unconditional mean
        theta_i_t = torch.div(w_i, (1 - B_i))
        theta_o_t = torch.div(w_o, (1 - B_o))
        theta_t = self.identify(torch.cat((theta_i_t, theta_o_t)))
        X_t = None
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            theta_T = torch.cat((theta_T, theta_t.unsqueeze(1)), 1)
            # The following lines must be the only diff in the loop between dgp and
            # filt func
            if X_T is not None:
                X_t = X_T[:, :, t]
            p_t = self.expMatBin(theta_t, delta=delta, X_t=X_t)
            A_t = self.sampleBin(p_t, delta=delta, X_t=X_t)
            A_T[:, :, t] = A_t.clone()
            ###
            theta_tp1, score = self.update_dynbin_par(A_t, theta_t, w, B, A, delta=delta, X_t=X_t)
            theta_t = theta_tp1.clone()

        return theta_T[:, 1:], A_T

    def sd_filt_bin(self, w, B, A, A_T, delta=None, X_T = None):
        """given the static parameters,  and the observations fiter the dynamical
        parameters with  score driven dynamics.
        """
        w_i, w_o = splitVec(w)
        B_i, B_o = splitVec(B)
        N = w_i.shape[0]

        T = A_T.shape[2]

        theta_T = torch.zeros((2 * N, 1))
        # initialize the time varying parameters to the unconditional mean
        theta_i_t = torch.div(w_i, (1 - B_i))
        theta_o_t = torch.div(w_o, (1 - B_o))
        theta_t = self.identify(torch.cat((theta_i_t, theta_o_t)))
        X_t = None
        for t in range(T):
            # observations at time tm1 are used to define the parameters at t
            theta_T = torch.cat((theta_T, theta_t.unsqueeze(1)), 1)
            if X_T is not None:
                X_t = X_T[:, :, t]
            # The following line must be the only diff in the loop between dgp and
            # filt func
            A_t = A_T[:, :, t]
            ###
            theta_tp1, score = self.update_dynbin_par(A_t, theta_t, w, B, A, delta=delta, X_t=X_t)
            theta_t = theta_tp1.clone()
            #print(torch.isnan(theta_tp1).sum())
        return theta_T[:, 1:]

    def loglike_sd_filt_bin(self, w, B, A, A_T, delta=None, X_T=None):
        """
        the loglikelihood of a sd filter for the delta model
        """
        theta_T = self.sd_filt_bin(w, B, A, A_T, delta=delta, X_T=X_T)
        T = theta_T.shape[1]
        logl_T = 0
        X_t = None
        for t in range(T):
            if X_T is not None:
                X_t = X_T[:, :, t]
            A_t = A_T[:, :, t]
            theta_t = theta_T[:, t]
            logl_T = logl_T + self.loglike_bin_t(A_t, theta_t, delta=delta, X_t=X_t)
        return logl_T


    def estimateBin_SD(self, A_T, opt_n=1, Steps=800, lRate=0.005, plot_flag=False,
                 B0=None, A0=None, W0=None):

        N = A_T.shape[0]
        if B0 is None:
            B, A = torch.tensor([0.7, 0.7]), torch.ones(2) * 0.0001
            wI, wO = 1 + torch.randn(N), torch.randn(N)
            w = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
        else:
            B = B0
            A = A0
            w = W0

        n_w = 2 * N
        n_A = n_B = B.shape[0]
        # set initial value for the parameters of the optimization
        unPar = torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))).clone().detach()
        unPar.requires_grad = True

        def obj_fun():
            reBA = self.un2re_BA_par(unPar[n_w:])
            return - self.loglike_sd_filt_bin(unPar[:n_w], reBA[:n_B], reBA[n_B:n_B + n_A], A_T)

        optimizers = [torch.optim.SGD([unPar], lr=lRate, nesterov=False),
                      torch.optim.Adam([unPar], lr=lRate),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.5, nesterov=True),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.7, nesterov=True)]
        legend = ["SGD", "Adam", "Nesterov 1", "Nesterov 2"]

        loss = obj_fun()
        print((loss.data, self.un2re_BA_par(unPar[n_w:]).data))
        diag = []
        for i in range(Steps):
            # define the loss
            loss = obj_fun()
            # set all gradients to zero
            optimizers[opt_n].zero_grad()
            # compute the gradients
            loss.backward(retain_graph=True)
            # take a step
            optimizers[opt_n].step()
            tmp = self.un2re_BA_par(unPar[n_w:])
            print((loss.data, tmp.data))
            diag.append(loss.item())
        if plot_flag:
            plt.figure()
            plt.plot(diag)
            plt.legend(legend[opt_n])
        w_est = unPar[:n_w].clone()
        re_BA_est = self.un2re_BA_par(unPar[n_w:])

        return w_est.clone(), re_BA_est[:n_B].clone(), re_BA_est[n_B:].clone(), diag


class dirBin1_X0_dynNet_SD(dirBin1_dynNet_SD):
    """
    This a class  for directed binarydynamical networks, modelled with
    one dynamical parameter per each node (hence the 1 in the name) each evolving with a Score Driven
    Dynamics, and one regressor for each link. the number of parameters associated with the regressors
    is flexible but for the moment we have a single parameter equal for all links
    """

    @staticmethod
    def regr_product_bin(delta, X_t):
        """
        Given a matrix of regressors and a vector of parameters obtain a matrix that will be sum to the matrix obtained
        from of nodes specific unrestricted dynamical paramters
        """
        if delta.shape[0] == 1:# one parameter for all links
            prod = delta * X_t
        elif delta.shape[0] == X_t.shape[0]:# one parameter for each node
            prod = torch.mul(delta + delta.unsqueeze(1), X_t)
        return prod

    def invPiMat(self, theta, delta, X_t):
        """
        given the vector of unrestricted parameters, return the matrix of
        of the inverses of odds obtained as products of exponentials
        In the paper we call it 1 / pi = exp(theta_i + theta_j)
        """
        theta_i, theta_o = splitVec(theta)
        return putZeroDiag(torch.mul(torch.mul(torch.exp(theta_i), torch.exp(theta_o).unsqueeze(1)),
                                     torch.exp(self.regr_product_bin(delta, X_t))))

    def estimate_ss_bin_t(self, A_t, X_t, delta, theta_0=None,
                          opt_steps=250, opt_n=1, lRate=0.1, print_flag=True, plot_flag=False, print_fun=None):
        """single snapshot Maximum logLikelihood estimate given delta"""
        if print_fun is None:
            degIO = strIO_from_mat(A_t)
            print_fun = lambda x: torch.mean(self.check_fitBin(x, degIO,  X_t=X_t, delta=delta))
        return self.estimate_ss_bin_t_MLE(A_t, theta_0=theta_0, fast_mle=False, X_t=X_t, delta=delta, opt_steps=opt_steps,
                              lRate=lRate, opt_n=opt_n, plot_flag=plot_flag, print_flag=print_flag, print_fun=print_fun)



    def estimate_delta_given_theta_T(self, A_T, X_T, theta_T, dim_delta, delta_0=None,
                        opt_steps=5000, opt_n=1, lRate=0.01, print_flag=True, plot_flag=False):
        """single snapshot Maximum logLikelihood estimate given delta"""

        T = A_T.shape[2]
        if delta_0 is None:
            delta_0 = torch.zeros(dim_delta)

        unPar_0 = delta_0.clone().detach()

        # the likelihood for delta is the sum of all the single snapshots likelihoods, given theta_T
        def obj_fun(unPar):
            logl_T=0
            for t in range(T):
                logl_T = logl_T + self.loglike_bin_t(A_T[:, :, t], theta_T[:, t], X_t=X_T[:, :, t], delta=unPar)
            return - logl_T

        delta_est, diag = optim_torch(obj_fun, unPar_0, opt_steps=opt_steps, opt_n=opt_n, lRate=lRate,
                                      plot_flag=plot_flag, print_flag=print_flag)

        return delta_est, diag

    def ss_filt_bin(self, A_T, X_T, dim_delta, delta_0=None, theta_T0=None,
                  opt_large_steps=10, opt_steps_theta=500, opt_steps_delta=200, lRate_delta=0.01, lRate_theta=0.5,
                  print_flag_theta=False, print_flag_delta=False, plot_flag_delta=False, plot_flag_theta=False, t_plot=1):
        """
        Static sequence filter.
         return the sequence  of single snapshot estimates for theta and the corresponding estimate for delta
          to avoid one very large optimization alternate:
            1 estimate of sequences of  theta given delta
            2 estimate of delta given theta_T
        """
        T = A_T.shape[2]
        N = A_T.shape[0]
        if delta_0 is None:
            delta = torch.zeros(dim_delta)
        else:
            delta = delta_0.clone()
        if theta_T0 is None:
            theta_T = torch.zeros(2*N, T)
        else:
            theta_T = theta_T0.clone()
        diag = []
        for n in range(opt_large_steps):
            print(n)
            like_theta = 0
            for t in range(T):
                # estimate each theta_t given delta
                A_t = A_T[:, :, t]
                X_t = X_T[:, :, t]
                if (t==t_plot) and plot_flag_theta :
                    plot_flag_theta_t = True
                else:
                    plot_flag_theta_t = False

                theta_t, diag_theta_t = self.estimate_ss_bin_t(A_t, X_t=X_t, delta=delta.clone().detach(),
                                                         theta_0=theta_T[:, t],
                                                         opt_steps=opt_steps_theta,
                                                         lRate=lRate_theta,
                                                         print_flag=print_flag_theta, plot_flag=plot_flag_theta_t)


                theta_t = self.identify(theta_t)
                theta_T[:, t] = theta_t

            diag.append(self.like_seq(A_T, theta_T, X_T=X_T, delta=delta).item())
            # estimate delta given theta_T
            delta, diag_delta = self.estimate_delta_given_theta_T(A_T, X_T, theta_T.clone().detach(), dim_delta, delta_0=delta,
                                                             lRate=lRate_delta,     opt_steps=opt_steps_delta,
                                                                 print_flag=print_flag_delta, plot_flag=plot_flag_delta)

            diag.append(self.like_seq(A_T, theta_T, X_T=X_T, delta=delta).item())

        return theta_T, delta, diag

    def estimateBin_SD(self, A_T, X_T, dim_delta, opt_n=1, Steps=800, lRate=0.005, plot_flag=False,
                 B0=None, A0=None, W0=None, delta0=None):
        N = A_T.shape[0]
        if B0 is None:
            B, A = torch.tensor([0.7, 0.7]), torch.ones(2) * 0.0001
            wI, wO = 1 + torch.randn(N), torch.randn(N)
            w = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
            delta = torch.zeros(dim_delta)
        else:
            B = B0
            A = A0
            w = W0
            delta=delta0

        n_w = 2 * N
        n_A = n_B = B.shape[0]
        # set initial value for the parameters of the optimization
        unPar = torch.cat((torch.cat((w, self.re2un_BA_par(torch.cat((B, A))))), delta)).clone().detach()
        unPar.requires_grad = True

        def obj_fun():
            reBA = self.un2re_BA_par(unPar[n_w:n_w + n_B + n_A])
            return - self.loglike_sd_filt_bin(unPar[:n_w], reBA[:n_B], reBA[n_B:], A_T,
                                            delta=unPar[n_w + n_B + n_A:], X_T=X_T)

        optimizers = [torch.optim.SGD([unPar], lr=lRate, nesterov=False),
                      torch.optim.Adam([unPar], lr=lRate),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.5, nesterov=True),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.7, nesterov=True)]
        legend = ["SGD", "Adam", "Nesterov 1", "Nesterov 2"]

        loss = obj_fun()
        print((loss.data, self.un2re_BA_par(unPar[n_w:n_w + n_B + n_A]).data, unPar[n_w + n_B + n_A:].data))
        diag = []
        for i in range(Steps):
            # define the loss
            loss = obj_fun()
            # set all gradients to zero
            optimizers[opt_n].zero_grad()
            # compute the gradients
            loss.backward(retain_graph=True)
            # take a step
            optimizers[opt_n].step()
            tmp = torch.cat((loss.unsqueeze(0), self.un2re_BA_par(unPar[n_w:n_w + n_B + n_A]), unPar[n_w + n_B + n_A:]))
            print((tmp.data))
            diag.append(loss.item())
        if plot_flag:
            plt.figure()
            plt.plot(diag)
            plt.legend(legend[opt_n])
        w_est = unPar[:n_w].clone()
        re_BA_est = self.un2re_BA_par(unPar[n_w:n_w + n_B + n_A])
        delta_est = unPar[n_w + n_B + n_A:]
        return w_est, re_BA_est[:n_B].clone(), re_BA_est[n_B:].clone(), delta_est.clone(), diag







