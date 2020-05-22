import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
import matplotlib.pyplot as plt
import numpy as np


# Purpose of this class is to define our neural_net architecture and the methods we need to train, save, and test our NN

class BasicDQN(nn.Module):

    def __init__(self, img_shape):
        super.__init__()
        in_feats = np.product(list(img_shape))
        # TODO: Finalize and implement final NN aritcheture to calculate q values
        self.fc1 = nn.Linear(in_features=in_feats, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=3)

    def forward(self, t):
        # TODO: Will need to update as architecture (above) changes
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


def get_current_QVals(policy_net: BasicDQN, states, actions):
    return policy_net.forward(states).gather(dim=1, index=actions.unsqueeze(-1))


def get_next_QVals(target_net, next_states):
    # TODO: get next q-values
    pass
