from vizdoom import *
import random, time
from DataStructures import *
from DQN import BasicDQN
import DQN
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

