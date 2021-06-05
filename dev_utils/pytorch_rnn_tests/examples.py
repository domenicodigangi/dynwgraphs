
import torch 
from torch import nn
import torch.nn.functional as F

nNodes = 30
nInput = nNodes**2
T = 100
AT = torch.randn(nNodes, nNodes, T)
input = torch.reshape(AT, (T, 1, -1))
input.size()
rnn = nn.RNN(nInput, nNodes, 2)

h0 = torch.randn(2, 1, nNodes)
output, hn = rnn(input, h0)


def update_score_driven_latent(f_t, obs_t, scaled_grad_fun, )


class scoreDrivenUpdateDiagPar(nn.Module, grad_fun, scaling_fun, nSdPar):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(nSdPar))
        self.B = nn.Parameter(torch.zeros(10))
        self.A = nn.Parameter(torch.zeros(10))

    def forward(self, f_t, obs_t):

        g_t = grad_fun(f_t, obs_t)
        s_t =  scaled_grad_fun(f_t, obs_t)

        f_tp1 = self.W +  self.B @ f_t + self.A @ 

        return f_tp1

class RnnScoreDriven(nn.Module, nSDPar):
    def __init__(self):
        super(RnnScoreDriven, self).__init__()
        self.rnn_linear_AR = nn.Linear(nSDPar, nSDPar)

    def forward(self, inp, latent):
        latent_AR_updated = self.rnn_linear_AR(latent)
        output = latent_AR_updated +  A @ score(latent, )
        return output


inp = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)

rnn = RnnWith2HiddenSizesModel()

output = RnnWith2HiddenSizesModel()(inp, (h0, c0))