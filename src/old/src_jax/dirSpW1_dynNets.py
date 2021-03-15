
#import autograd.numpy as np
#from autograd import grad, jacobian, hessian
import jax
import jax.numpy as npj
from jax import grad, hessian, jit
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import scipy as sp



from utils import splitVec, putZeroDiag,gen_test_net,degIO_from_mat,strIO_from_mat,npj_t,optim_jax,tic,toc
#% Zero Augmented Static model funcitons


def cond_exp_Y(phi,alpha = 1,p_mat=None):
    """
    given the parameters related with the weights, compute the conditional expectation of matrix Y.
    the conditioning is on each matrix element being greater than zero
    """
    phi_i,phi_o = splitVec(phi ) 
    phi_o_trasp = npj.reshape(phi_o,[-1,1])
#    if p_mat is None:
    EY_cnd_mat = alpha * npj.exp(phi_i + phi_o_trasp)
#    else:
#        EY_cnd_mat = alpha * npj.divide(npj.exp(phi_i + phi_o),p_mat)
    return EY_cnd_mat


def sample_ZA_gam(p_mat,EYcond_mat, alpha=1):
    """
    given the matrix of probabilities of observing a link, and the matrix of 
    expected links' weights, conditional on the link being present,
    sample once the ZA gamma distribution 
    """
    A_mat = np.array( np.random.binomial(1,p_mat) ,dtype=bool)
    mu_mat = np.divide(EYcond_mat,alpha)
    
    ZAgamma_s = np.random.gamma(alpha, mu_mat )
    ZAgamma_s[np.logical_not(A_mat)] = 0
    return npj.array(ZAgamma_s)    
 
def loglike_t(Y_mat,p_mat,phi,alpha):
    """ The log likelihood of the zero augmented gamma network model as a funciton of the
    observations and the matrices of parameters p_mat (pi = p/(1+p), where pi is the prob of being
    nnz), and mu_mat E[Y_ij] = mu_ij
    """
    A_mat = npj.greater(Y_mat,0)
#    print((A_mat.shape,npj.sum(A_mat)))
    mu_mat = cond_exp_Y(phi,alpha=alpha)

    tmp1 = npj.log(p_mat[A_mat]) + (alpha- 1) * npj.log(Y_mat[A_mat]) \
            - alpha * npj.log(mu_mat[A_mat]) - npj.divide(Y_mat[A_mat],mu_mat[A_mat])

    tmp2 = - alpha  *npj.sum(A_mat)* sp.special.loggamma(alpha)
    loglike =  - npj.sum(npj.log(1 + p_mat)) + npj.sum( tmp1) + tmp2
    return loglike



def loglike_w_t_non_jit(Y_mat,phi,alpha=1):
    """ only the part that depends on phi and alpha parameters
    """
    A_mat = npj.greater(Y_mat,0)
    mu_mat = cond_exp_Y(phi,alpha=alpha)
    #divide the computation of the loglikelihood in 4 pieces
    tmp =   (alpha- 1) * npj.sum( npj.log(Y_mat[A_mat]) )  
    tmp1 = - npj.sum(A_mat)* jax.scipy.special.gammaln(alpha)
    tmp2 = - alpha * npj.sum( npj.log(mu_mat[A_mat]) )
    tmp3 = - npj.sum( npj.divide(Y_mat[A_mat],mu_mat[A_mat]) ) 
    return tmp  +  tmp1 + tmp2 + tmp3

def loglike_w_t_jittable_(Y_mat,phi,alpha=1):
    """ only the part that depends on phi and alpha parameters
        and avoids indexing by boolean unknown in advance
        it can be improoved, as soon as some numpy functions (e.g. nonzero)
        are implemented in jax
    """
    A_mat = npj.greater(Y_mat,0)
    mu_mat = cond_exp_Y(phi,alpha=alpha)

    #divide the computation of the loglikelihood in 4 pieces        
    # log(0) * 0 = nan, using then nansum is equivalent to bool inds
    tmp =  (alpha - 1) *npj.nansum(npj.multiply(npj.log(Y_mat),A_mat) )   
    tmp1 = 0#- npj.sum(A_mat)* jax.scipy.special.gammaln(alpha)    
    tmp2 = - alpha* npj.nansum( npj.multiply(  npj.log(mu_mat) ,A_mat))  
    tmp3 = - npj.nansum(npj.multiply(npj.divide(Y_mat,mu_mat) ,A_mat) )
    return tmp +  tmp1 + tmp2 + tmp3

loglike_w_t_jittable = jit(loglike_w_t_jittable_)
loglike_w_t = loglike_w_t_jittable
#    
def estimate_cs_ZA_wt(Y_mat,p_mat=None, phi_0 = None,alpha=1,print_info = False):     
    N = Y_mat.shape[0]
    if p_mat is None:
        p_mat = npj.zeros((N,N))
    def obj_fun(phi):
        return  -loglike_w_t_jittable(Y_mat, phi,alpha=alpha)    
    if phi_0 is None:
        phi_0  = -0.01 * npj.ones(2*N)   
    #print(obj_fun((phi_0)))
    
    return optim_jax(obj_fun,phi_0,print_info = print_info)
 

#print(loglike_w_t_jittable(Y_t,phi_0),loglike_w_t_non_jit(Y_t,phi_0))
#
def identify_(phi):
    """ enforce an identification condition on the parameters of ZA gamma net
    model
    """
    # set the first in parameter to zero
    delta = phi[0]
    
    phi_i,phi_o = splitVec(phi)
    phi_i_out = phi_i - delta
    phi_o_out = phi_o + delta
#    phi = jax.ops.index_add(phi, jax.ops.index[N:2*N],delta   )
#    phi = jax.ops.index_add(phi, jax.ops.index[0:N], -delta   )
##    phi[:N] = phi[:N] -phi[0]
#    phi[N:2*N] = phi[N:2*N] + phi[0]    
    return npj.concatenate((phi_i_out,phi_o_out))
identify = jit(identify_)




 
def ss_filt_w(Y_T):
    """
    Static sequence filter.
     return the sequence  of single snapshot estimates
    """
    T = Y_T.shape[2]
    N = Y_T.shape[0]
    phi_T = npj.zeros((2*N,T))
    for t in range(T):
        print(t)
        Y_t = Y_T[:,:,t]
        phi_t,info =  estimate_cs_ZA_wt(Y_t)
        phi_t = identify(phi_t)
        phi_T = jax.ops.index_add(phi_T, jax.ops.index[:,t] , phi_t) 
    return phi_T
        

#%Score Driven Dynamics' Functions
    
def resc_score(s_i,s_o,Y_mat,alpha=1):
    return s_i,s_o



def un2re_BA_par(BA_un):
    B_un,A_un = splitVec(BA_un)
    exp_B = npj.exp(B_un)
    return npj.concatenate( (npj.divide(exp_B,(1+exp_B)) ,npj.exp(A_un)) )

def re2un_BA_par(BA_re):
    B_re,A_re = splitVec(BA_re)
    return npj.concatenate( (npj.log( npj.divide(B_re,1-B_re)) , npj.log(A_re)) )
                        

 

def score_w_t_(Y_t,phi_t,alpha=1):
    """ given the observations and the ZA gamma parameters (i.e. the cond mean
    matrix and the alpha par), return the score of the distribution
    """
    A_t = npj.array(Y_t, dtype=bool)
    deg_i = npj.sum(A_t,axis=0)
    deg_o = npj.sum(A_t,axis = 1)
    ratio_mat = npj.divide(Y_t,cond_exp_Y(phi_t,alpha = alpha))
    score_i = ( npj.sum(ratio_mat,axis = 0) -  deg_i ) * alpha 
    score_o = ( npj.sum(ratio_mat,axis = 1) -  deg_o ) * alpha
    
    return score_i,score_o
score_w_t = jit(score_w_t_)

def score_w_t_AD_(Y_t,phi_t,alpha=1):
    """ given the observations and the ZA gamma parameters (i.e. the cond mean
    matrix and the alpha par), return the score of the distribution
    """   
    # conpute the score with AD using either Autograd or jax
#    score_i,score_o = splitVec( grad(loglike_w_t,1)(Y_t, phi,alpha=alpha))
    def obj_fun(phi): 
        return  loglike_w_t(Y_t,phi,alpha = alpha)
    score_i,score_o = splitVec( grad(obj_fun)( phi_t) )
    return score_i,score_o
score_w_t_AD = jit(score_w_t_AD_)


def update_dynw_par_(Y_t,phi_t, w,B,A,alpha = 1):
    """ score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
    """
    phi_t = identify(phi_t)
    phi_i_t,phi_o_t = phi_t[:N],phi_t[N:2*N]
    w_i,w_o = npj.split(w,2)
    B_i,B_o = npj.split(B,2)
    A_i,A_o = npj.split(A,2)
    s_i,s_o = score_w_t_AD(Y_t,phi_t)
#    s_i_2,s_o_2 = score_w_t(Y_t,phi_t)
#    print(npj.sum(s_i-s_i_2) + npj.sum(s_o-s_o_2))
#    print(npj.mean(s_i),npj.mean(s_o))
    ss_i,ss_o = resc_score(s_i,s_o,Y_t,alpha=alpha)
    phi_i_tp1 = w_i + npj.multiply(B_i,phi_i_t) + npj.multiply(A_i,ss_i)
    phi_o_tp1 = w_o + npj.multiply(B_o,phi_o_t) + npj.multiply(A_o,ss_o)
    phi_tp1 = npj.concatenate((phi_i_tp1,phi_o_tp1))
    
    return  identify(phi_tp1)
update_dynw_par =  jit(update_dynw_par_)

def sd_dgp_w(w,B,A,p_T,alpha = 1):
    """given the static parameters, sample the dgp with a score driven dynamics. 
    equal to the filter except sample the observations for each t.
    """
    w_i,w_o = npj.split(w,2)
    B_i,B_o = npj.split(B,2)
    A_i,A_o = npj.split(A,2)
    N = w_i.shape[0]

    T = p_T.shape[2]

    Y_T = npj.zeros((N,N,T))

    phi_T = npj.zeros((2*N,T))
    # initialize the time varying parameters to the unconditional mean
    phi_i_t = npj.divide(w_i,(1-B_i))
    phi_o_t = npj.divide(w_o,(1-B_o))
    phi_t = identify(npj.concatenate((phi_i_t,phi_o_t)))
    for t in range(T):
       #observations at time tm1 are used to define the parameters at t  
        phi_T = jax.ops.index_add(phi_T, jax.ops.index[:,t] , phi_t)      
        # The following lines must be the only diff in the loop between dgp and 
        # filt func
        p_t = p_T[:,:,t]
        Y_t = sample_ZA_gam(p_t,cond_exp_Y(phi_t,alpha = alpha),alpha)
        Y_T = jax.ops.index_add(Y_T, jax.ops.index[:,:,t] , Y_t) 
        ###
        phi_tp1 = update_dynw_par(Y_t,phi_t, w,B,A,alpha = 1) 
        phi_t = phi_tp1  

    return phi_T,Y_T
    

def sd_filt_w_(w,B,A,p_T,Y_T,alpha = 1):
    """given the static parameters,  and the observations fiter the dynamical
    parameters with  score driven dynamics. 
    """
    w_i,w_o = npj.split(w,2)
    B_i,B_o = npj.split(B,2)
    A_i,A_o = npj.split(A,2)
    N = w_i.shape[0]

    T = p_T.shape[2]
        
    phi_T = npj.zeros((2*N,T))
    # initialize the time varying parameters to the unconditional mean
    phi_i_t = npj.divide(w_i,(1-B_i))
    phi_o_t = npj.divide(w_o,(1-B_o))
    phi_t = identify(npj.concatenate((phi_i_t,phi_o_t)))
    
    for t in range(T):
        #observations at time tm1 are used to define the parameters at t  
        phi_T = jax.ops.index_add(phi_T, jax.ops.index[:,t] , phi_t) 
           
        # The following line must be the only diff in the loop between dgp and 
        # filt func
        Y_t = Y_T[:,:,t]
        ###
        phi_tp1 = update_dynw_par(Y_t,phi_t, w,B,A,alpha = 1) 
        phi_t = phi_tp1  
        
    return phi_T
sd_filt_w =jit(sd_filt_w_)

def seq_bin(p_mat,Y_T = None,T = None):
    if (T is None) and (Y_T is None):
        raise Exception("seq must be used as a dgp or as a filter")
        
    return npj.moveaxis(npj.tile(p_mat,(T,1,1)),0,2)
#%% Define test variables
N = 20    
T = 250
p_const = 0.5
alpha = 1
p_t = p_mat_unif = npj.ones((N,N)) * p_const
p_T = seq_bin(p_mat_unif,T=T)

B,A =npj.array([0.9,0.8]),npj.ones(2)*0.01
wI,wO = 1,1
w = npj.concatenate((np.ones(N)*wI, np.ones(N)*wO))

density = 0.5
Y_t = gen_test_net(N,density)
phi_T,Y_T = sd_dgp_w(w,B,A,p_T)
 
#% Goal: Sample a Score Driven DGP with uniform constant Pmat

t_test = 0
w_i,w_o = npj.split(w,2)
B_i,B_o = npj.split(B,2)

um_i,um_o = w_i/(1-B_i),w_o/(1-B_o)
um = npj.concatenate((um_i,um_o))
A_i,A_o = npj.split(A,2)
phi_i_t = npj.divide(w_i,(1-B_i))
phi_o_t = npj.divide(w_o,(1-B_o))
phi_t = npj.concatenate((phi_i_t,phi_o_t))
#phi_t = phi_T[:,t_test]
Y_t = Y_T[:,:,t_test]
Y_t1 =  sample_ZA_gam(p_t,cond_exp_Y(phi_t,alpha = alpha),alpha) #
#score_w_t_AD(1Y_t,phi_t)
#score_w_t(Y_t,phi_t)
#cond_exp_Y(phi_t)
#update_dynw_par( Y_t,phi_t,w,B,A,alpha = 1)

density = 0.5
Y_t = gen_test_net(N,density)
phi_T,Y_T = sd_dgp_w(w,B,A,p_T)
tic()
sd_filt_w_(w,B,A,p_T,Y_T)
toc()
#%% Goal estimate a SD filter

def loglike_sd_filt_w_old(w,B,A,p_T,Y_T,alpha = 1):
    """ the loglikelihood of a sd filter for the parameters driving the 
    conditional mean, as a function of the static score driven dynamics
    parameters and alpha
    """
    phi_T = sd_filt_w_(w,B,A,p_T,Y_T,alpha = alpha)
    T = phi_T.shape[1]
    logl_T = 0
    for t in range(T):
        Y_t = Y_T[:,:,t]
        phi_t = phi_T[:,t]
        logl_T = logl_T + loglike_w_t(Y_t,phi_t,alpha = alpha)
    return logl_T

def loglike_sd_filt_w_(w,B,A,p_T,Y_T,alpha = 1):
    """ the loglikelihood of a sd filter for the parameters driving the 
    conditional mean, as a function of the static score driven dynamics
    parameters and alpha 
    """
    w_i,w_o = npj.split(w,2)
    B_i,B_o = npj.split(B,2)
    A_i,A_o = npj.split(A,2)
    N = w_i.shape[0]
    T = p_T.shape[2]        
    # initialize the time varying parameters to the unconditional mean
    phi_i_t = npj.divide(w_i,(1-B_i))
    phi_o_t = npj.divide(w_o,(1-B_o))
    phi_t = identify(npj.concatenate((phi_i_t,phi_o_t)))
    logl_T = 0
    for t in range(T):
        Y_t = Y_T[:,:,t]
        logl_T = logl_T + loglike_w_t(Y_t,phi_t,alpha = alpha)
        phi_tp1 = update_dynw_par(Y_t,phi_t, w,B,A,alpha = 1) 
        phi_t = phi_tp1  
    return logl_T
 
loglike_sd_filt_w = loglike_sd_filt_w_#jit(loglike_sd_filt_w_) 
print("Compiling obj_fun. It might take some time")
tic()
loglike_sd_filt_w(w,B,A,p_T,Y_T)
toc()
loglike_sd_filt_w_old(w,B,A,p_T,Y_T)


   
#%%


#%% GOAL: estimate the unconditional means as means of single snapshot estimates



#%% GOAL: given the unconditional means estimate B and A parameters via targeting
phi_ss_T = ss_filt_w(Y_T)
# assuming the unconditional means to be known in advance
um_i,um_o = splitVec(um)
wI,wO = 1,1
w = npj.concatenate((np.ones(N)*wI, np.ones(N)*wO))
def obj_fun_(vUnPar):
    
        B_re,A_re = splitVec(un2re_BA_par(vUnPar))
        B_re_i,B_re_o = splitVec(B_re)
        w = npj.concatenate((um_i*(1-B_re_i),um_o*(1-B_re_o)))
        return - loglike_sd_filt_w_(w,B_re,A_re,p_T,Y_T)

obj_fun = obj_fun_
jac = jax.jacfwd(obj_fun_)
hess = jax.jacfwd(jac)
N = Y_T.shape[0]
#Define the vector of static restricted parameters
B_0re,A_0re =npj.ones(2)*0.7,npj.ones(2)*0.1
vRePar_0 = npj.concatenate((B_0re,A_0re)) 
 
vUnPar_0 = re2un_BA_par( vRePar_0)   

print("Compiling obj_fun. It might take some time")
tic()
obj_fun(vUnPar_0)
toc()
print("Compiling jac. It might take some time")
tic()
jac(vUnPar_0)
toc()

print("Compiling hess. It might take some time")
tic()
hess(vUnPar_0)
toc()
#%
optBA,info = optim_jax(obj_fun,vUnPar_0,jac=jac,hess=hess,print_info=True)
un2re_BA_par(optBA)
#%%


#%%
def est_sd_filt_w(Y_T,p_T):
    T = Y_Y.shape[2]
    for t in range(T):
        Y_t = Y_T[:,:,t]
        
    return optim_jax(obj_fun,vUnPar_0)

out =est_sd_filt_w(Y_T,p_T)


















