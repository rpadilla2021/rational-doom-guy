import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
import numpy as np

# Purpose of this class is to define our neural_net architecture and the methods we need to train, save, and test our NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicDQN(nn.Module):

    def __init__(self, img_shape, output_actions):
        super().__init__()
        self.in_shape = img_shape

        self.actions = output_actions
        out_feats = len(output_actions)
        # TODO: Finalize and implement final NN aritcheture to calculate q values
        print("CREATING THE NET, INPUT FEATURES", img_shape, "      OUTPUT FEATURES ", out_feats)
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1)
        self.pool1 = nn.MaxPool2d((2, 2), padding=(0, 0), dilation=(1, 1))
        self.conv2 = nn.Conv2d(6, 2, 2, stride=1)

        f = self.get_dim_post_conv

        resize_dim = f(f(f(f(img_shape, self.conv1), self.pool1), self.conv2), self.pool1)
        print("After Convolutions Size: ", resize_dim)

        in_feats = np.product(list(resize_dim))
        self.fc_feats = in_feats

        self.fc1 = nn.Linear(in_features=in_feats, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.out = nn.Linear(in_features=8, out_features=out_feats)

    @staticmethod
    def get_dim_post_conv(img_shape, conv_layer):
        if len(img_shape) == 2:
            img_shape = (1, img_shape[0], img_shape[1])

        in_channels, h_in, w_in = img_shape

        assert img_shape[0] == in_channels
        # print(conv_layer.padding)
        h_out = (h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / \
                conv_layer.stride[0] + 1
        w_out = (w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / \
                conv_layer.stride[1] + 1
        # print(h_out, w_out)
        h_out, w_out = int(np.floor(h_out)), int(np.floor(w_out))

        if not hasattr(conv_layer, 'out_channels'):
            result = in_channels, h_out, w_out
        else:
            result = conv_layer.out_channels, h_out, w_out

        # print(result)
        return result

    def forward(self, t):
        # TODO: Will need to update as architecture (above) changes
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).unsqueeze(0).to(device)
        t = t.unsqueeze(1)

        t = F.relu(self.conv1(t))
        t = self.pool1(t)
        t = F.relu(self.conv2(t))
        t = self.pool1(t)
        # print(t.shape)
        t = t.view(-1, self.fc_feats)
        # print(t.shape)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

    def select_best_action(self, state, show=False):
        if len(state.shape) <= 2:
            state = state.unsqueeze(0)
        with torch.no_grad():
            raw_vals = self.forward(state)
            if show:
                print(raw_vals)

            # print(raw_vals)
            result = raw_vals.argmax(dim=1).item()  # Exploitation
            result = self.actions[result]
            if show:
                print(result, "\n")
            return result


class HealthDQN(nn.Module):

    def __init__(self, img_shape, output_actions):
        super().__init__()
        self.in_shape = img_shape

        self.actions = output_actions
        out_feats = len(output_actions)
        # TODO: Finalize and implement final NN aritcheture to calculate q values
        print("CREATING THE NET, INPUT FEATURES", img_shape, "      OUTPUT FEATURES ", out_feats)
        self.conv1 = nn.Conv2d(1, 7, 5, stride=1)
        self.pool1 = nn.MaxPool2d((2, 2), padding=(0, 0), dilation=(1, 1))
        self.conv2 = nn.Conv2d(7, 4, 3, stride=1)
        self.conv3 = nn.Conv2d(4, 2, 2, stride=1)
        f = self.get_dim_post_conv

        resize_dim = f(f(f(f(f(img_shape, self.conv1), self.pool1), self.conv2), self.pool1), self.conv3)
        print("After Convolutions Size: ", resize_dim)

        in_feats = np.product(list(resize_dim))
        self.fc_feats = in_feats

        self.fc1 = nn.Linear(in_features=in_feats, out_features=16)
        self.lstm1 = nn.LSTM(16, 16)
        self.fc2 = nn.Linear(in_features=16, out_features=10)
        # self.drop1 = nn.Dropout(0.1)
        self.out = nn.Linear(in_features=10, out_features=out_feats)

    @staticmethod
    def get_dim_post_conv(img_shape, conv_layer):
        if len(img_shape) == 2:
            img_shape = (1, img_shape[0], img_shape[1])

        in_channels, h_in, w_in = img_shape

        assert img_shape[0] == in_channels
        # print(conv_layer.padding)
        h_out = (h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / \
                conv_layer.stride[0] + 1
        w_out = (w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / \
                conv_layer.stride[1] + 1
        # print(h_out, w_out)
        h_out, w_out = int(np.floor(h_out)), int(np.floor(w_out))

        if not hasattr(conv_layer, 'out_channels'):
            result = in_channels, h_out, w_out
        else:
            result = conv_layer.out_channels, h_out, w_out

        # print(result)
        return result

    def forward(self, t):
        # TODO: Will need to update as architecture (above) changes
        if type(t) == np.ndarray:
            t = torch.from_numpy(t).unsqueeze(0).to(device)
        t = t.unsqueeze(1)

        t = F.relu(self.conv1(t))
        t = self.pool1(t)
        t = F.relu(self.conv2(t))
        t = self.pool1(t)
        t = F.relu(self.conv3(t))
        # print(t.shape)
        t = t.view(-1, self.fc_feats)
        t = F.relu(self.fc1(t))
        # print(t.unsqueeze(-1).shape)
        t, _ = self.lstm1(t.unsqueeze(0))
        t = t.squeeze(0)
        # t = self.drop1(t)
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

    def select_best_action(self, state, show=False):
        if len(state.shape) <= 2:
            state = state.unsqueeze(0)
        with torch.no_grad():
            raw_vals = self.forward(state)
            if show:
                print(raw_vals)

            result = raw_vals.argmax(dim=1).item()  # Exploitation
            result = self.actions[result]
            if show:
                print(result, "\n")
            return result


def get_current_QVals(policy_net: nn.Module, states: torch.Tensor, actions: torch.Tensor):
    raw_outputs = policy_net(states).to(device)
    # print("raw out shape", raw_outputs.shape)
    actions = torch.argmax(actions, dim=1).to(device)
    # print("index shape", actions.shape)
    result = raw_outputs.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    # print("result shape", result.shape)
    # print(result)
    return result


def get_next_QVals(target_net: nn.Module, next_states: torch.Tensor):
    # TODO: get next q-values
    terminal_state_locs = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_terminal_state_locs = (terminal_state_locs == False)
    non_terminal_states = next_states[non_terminal_state_locs]
    results = torch.zeros(next_states.shape[0]).to(device)
    results[non_terminal_state_locs] = target_net.forward(non_terminal_states).max(dim=1)[0].detach()
    return results
