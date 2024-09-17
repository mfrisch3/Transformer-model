import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
