import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
import matplotlib.pyplot as plt


# Purpose of this class is to define our neural_net architecture and the methods we need to train, save, and test our NN

class BasicDQN(nn.Module):

    def __init__(self, img_height, img_width):
        super.__init__()
        # TODO: Finalize and implement final NN aritcheture to calculate q values
        self.fc1 = nn.Linear(in_features=img_height * img_width, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.out = nn.Linear(in_features=10, 3)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.softmax(self.out(t))
        return t
