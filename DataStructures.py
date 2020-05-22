# Purpose of this class is to create a data structue that can hould our memories efficiently temporarily in memory
# This will allow us to abstract away many of the nitty gritty details of storing experiences
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, max_capacity):
        self.q = deque()
        self.cap = max_capacity
        self.push_count = 0

    def push(self, item):
        if len(self.q) < self.cap:
            self.q.append(item)
        else:
            self.q.pop()
            self.q.append(item)

        self.push_count += 1

    def can_sample(self, expected_size):
        return len(self.q) >= expected_size

    def sample(self, size=1):
        if not self.can_sample(size):
            assert False, "Sample size too large to extract from replay memory"
        return np.random.choice(list(self.q), size, replace=False)

    def sample_tensors(self, size=1):
        exp_seperate = self.sample(size)
        batch_exp = Experience(*zip(*exp_seperate))

        s = torch.cat(batch_exp.state)
        a = torch.cat(batch_exp.action)
        r = torch.cat(batch_exp.reward)
        s_prime = torch.cat(batch_exp.next_state)

        return s, a, s_prime, r


class Explorer:
    def __init__(self, epsilon_initial, epsilon_final, decay_rate):
        self.i, self.f, self.rate = epsilon_initial, epsilon_final, decay_rate
        self.step_ctr = 0

    def curr_epsilon(self):
        result = self.f + (self.i - self.f) * np.e ** (-1 * self.step_ctr * self.rate)
        self.step_ctr += 1
        return result

    def curr_epsilon_step(self, step):
        return self.f + (self.i - self.f) * np.e ** (-1 * step * self.rate)
