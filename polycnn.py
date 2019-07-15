"""
polycnn.py
POLYnomial Convolutional Neural Netowrk for image classification

Author: Joseph Daws Jr
Date: July 11, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import quadratic polynomial net
from qupo import Qupo

class Polycnn(nn.Module):
    def __init__(self,L=4):
        super(Polycnn, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # replace the linear layers with a qupo net
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        N_in = 16*5*5
        N_out = 10 
        self.polymap = Qupo(L,N_in,N_out,a=-1,b=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.polymap(x)
        return x

    def poly_init(self):
        with torch.no_grad():
            self.polymap.poly_init()
