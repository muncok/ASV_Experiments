import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import math


def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)


def num_flat_features(x):
    '''
    flatten the activations in the way conv to fc
    :param x:
    :return:
    '''
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        # normalization
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)


class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("saved to {}".format(filename))

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print("loaded from {}".format(filename))

    def load_partial(self, filename):
        to_state = self.state_dict()
        from_state = torch.load(filename)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        valid_state.pop('output.weight', None)
        valid_state.pop('output.bias', None)
        to_state.update(valid_state)
        self.load_state_dict(to_state)
        assert(len(valid_state) > 0)
        print("loaded from {}".format(filename))
