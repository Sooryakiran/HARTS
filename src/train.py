import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from harts import *
from model import *

"""
Hyper parameters

"""
EPOCHS         = 10
LEARNING_RATES = [1e-2, 1e-3]
MOMENTUM       = 0.5
BATCH_SIZE     = 512
LOG_INTERVAL   = 5

class CustomTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = x*2 - 1
        return x.view(-1)

TRAIN_LOADER = torch.utils.data.DataLoader(torchvision.datasets.MNIST('.', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           CustomTransform()])),
                                           batch_size=BATCH_SIZE, shuffle=True)

TEST_LOADER = torch.utils.data.DataLoader(torchvision.datasets.MNIST('.', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          CustomTransform()])),
                                          batch_size=BATCH_SIZE, shuffle=True)

def train(network, epoch):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("TRAINING ON GPU")

    network = network.to(device)
    criterion = nn.CrossEntropyLoss()
    if epoch < len(LEARNING_RATES):
        lr = LEARNING_RATES[epoch]
    else:
        lr = LEARNING_RATES[-1]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for index, (data, target) in enumerate(TRAIN_LOADER):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if index%LOG_INTERVAL == 0:
            pred = output.data.max(1, keepdim=True)[1]
            acc = pred.eq(target.data.view_as(pred)).sum()*1.0/BATCH_SIZE
        
            print("EPOCH: %d: BATCH (%d), \tLOSS = %0.3f, \tACC:%0.2f" %(epoch, index, loss.item(), acc.item()))

def test(network):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("TESTING ON GPU")

    network = network.to(device)
    criterion = nn.CrossEntropyLoss()
    

    for index, (data, target) in enumerate(TEST_LOADER):
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = criterion(output, target)
        loss.backward()

        if index%LOG_INTERVAL == 0:
            pred = output.data.max(1, keepdim=True)[1]
            acc = pred.eq(target.data.view_as(pred)).sum()*1.0/BATCH_SIZE
        
            print("BATCH (%d), \tLOSS = %0.3f, \tACC:%0.2f" %(index, loss.item(), acc.item()))


def train_multiple_epochs(network):
    for epoch in range(EPOCHS):
        train(network, epoch)

network = Network()
train_multiple_epochs(network)

torch.save(network, "saved_model.pth")
test(network)

