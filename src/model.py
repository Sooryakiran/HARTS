import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from harts import *

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.select_1 = SELECT(in_neurons = 784, out_neurons = 256)
        self.bin_1    = BinLayer(in_neurons = 256)

        self.select_2 = SELECT(in_neurons = 256*256, out_neurons = 256)
        self.not_2    = NotLayer(in_neurons = 256)
        self.bin_2    = BinLayer(in_neurons = 256)

        self.select_3 = SELECT(in_neurons = 256*256, out_neurons = 128)
        self.not_3    = NotLayer(in_neurons = 128)
        self.bin_3    = BinLayer(in_neurons = 128)

        self.select_4 = SELECT(in_neurons = 128*128, out_neurons= 64)
        self.not_4    = NotLayer(in_neurons = 64)
        self.bin_4    = BinLayer(in_neurons = 64)

        self.select_5 = SELECT(in_neurons = 64*64, out_neurons = 32)
        self.not_5    = NotLayer(in_neurons = 32)
        self.bin_5    = BinLayer(in_neurons = 32)

        self.select_6 = SELECT(in_neurons = 32*32, out_neurons = 10)

    def forward(self, x):
        x = self.select_1(x)
        x = self.bin_1(x)

        x = self.select_2(x)
        x = self.not_2(x)
        x = self.bin_2(x)

        x = self.select_3(x)
        x = self.not_3(x)
        x = self.bin_3(x)

        x = self.select_4(x)
        x = self.not_4(x)
        x = self.bin_4(x)

        x = self.select_5(x)
        x = self.not_5(x)
        x = self.bin_5(x)
        x = self.select_6(x)
        return x