import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinLayer(nn.Module):
    """
    Class BinLayer

    """
    def __init__(self, in_neurons, temp = 0.01):
        """
        The class constructor

        @param in_neurons : int, bit size of input
        @param temp       : float, temperature of softmax

        Let us say that there are n bits as input. We have n x n possible input pairs.
        However this reduces by nearly a factor of 2 for commutative bitwise operations.
        In this module, we take the entire n x n as search domain, so that we can support
        bitwise operations that are not commutative in future.

        Assume we have an input bit vector x. We compute 3 output matrices,
        OR  (x', x) 
        AND (x', x)
        XOR (x', x)
        Where x' is the transpose of x.

        During forward pass, we choose one among theses 3 matrices element wise to
        create an nxn matrix. The choice is done by doing a softmax over the stored weights.
        The weight matrix is updated using backpropagation to improve the choice.

        """

        super(BinLayer, self).__init__()
        self.in_neurons = in_neurons

        self.and_ = AND()
        self.or_  = OR()
        self.xor_ = XOR()
        
        self.weights = torch.nn.Parameter(data = torch.Tensor(in_neurons*in_neurons, 3), requires_grad = True)
        self.softmax = nn.Softmax(dim = -1)
        self.temp    = temp

        self.weights.data.uniform_(-1, 1)

    def forward(self, x):
        """
        The forward pass

        @param x: input bit vector

        """
        and_output = self.and_(x).unsqueeze(-1)
        or_output  = self.or_(x).unsqueeze(-1)
        xor_output = self.xor_(x).unsqueeze(-1)

        soft_weights   = self.softmax(self.weights/self.temp)
        massive_input  = torch.cat([and_output, or_output, xor_output], -1)
        massive_weight = soft_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        massive_output = torch.mul(massive_input, massive_weight)

        return torch.sum(massive_output, axis = -1)

class SELECT(nn.Module):
    """
    The SELECT class

    """
    def __init__(self, in_neurons, out_neurons, temp = 0.01):
        """
        Since the BinLayer defined above is not scalable, i.e it produces nxn outputs
        for an input of size n, we define a module that subsamples the input. The sub
        sampling choices are also trainable using backpropagation, since we use 
        softmax over the stored weights to perform the choice.

        @param in_neurons  : int, input bit vector length
        @param out_neurons : int, subsampled output bit vector length

        """
        super(SELECT, self).__init__()
        self.weights = torch.nn.Parameter(data = torch.Tensor(in_neurons, out_neurons), requires_grad = True)
        self.softmax = nn.Softmax(dim = 0)
        self.temp    = temp

        self.weights.data.uniform_(-1, 1)

    def forward(self, x):
        """
        Selects the most suitable top out_neurons

        """
        soft_weights = self.softmax(self.weights/self.temp)
        return x @ soft_weights

class NotLayer(nn.Module):
    """
    The NOT layer

    """
    def __init__(self, in_neurons, temp = 0.01):
        """
        The NOT layer can be used as a possible 'activation function'. This layer 
        chooses whether to put a NOT gate or not on each elements in the input bit
        vector. Like above, the choice is differentiable.

        @param in_neurons : int, input bit vector length
        @param temp       : float, softmax temperature

        """
        super(NotLayer, self).__init__()
        self.weights = torch.nn.Parameter(data = torch.Tensor(in_neurons), requires_grad = True)
        self.sigmoid = nn.Sigmoid()
        self.temp    = temp

        self.weights.data.uniform_(-1, 1)

    def forward(self, x):
        """
        The forward pass

        """
        x_in            = x
        x_compliment_in = -x_in
        soft_weights    = self.sigmoid(self.weights/self.temp)
        return torch.mul(x_in, soft_weights) + torch.mul(x_compliment_in, 1 - selft_weights)

class XOR(nn.Module):
    """
    The XOR module

    """
    def __init__(self):
        super(XOR, self).__init__()
    
    def forward(self, x):
        x      = x.unsqueeze(-1)
        output = -x @ torch.transpose(x, 1, 2)

        return output.view(x.size(0), -1)


class AND(nn.Module):
    """
    The AND module

    """
    def __init__(self):
        super(AND, self).__init__()
        self.xor = XOR();

    def forward(self, x):
        x = (x+1)/2 
        return - self.xor(x)*2 -1

class OR(nn.Module):
    """
    The OR module
    
    """
    def __init__(self):
        super(OR, self).__init__()
        self.xor  =  XOR()
        self.and_ = AND();

    def forward(self, x):
        return self.xor(x) + self.and_(x) + 1