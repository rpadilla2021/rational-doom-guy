from vizdoom import *
import random, time
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import namedtuple
from DataStructures import *
from DQN import BasicDQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


def print_game_state(gameState, notebook=False):
    print("Number:", gameState.number, "\t Tic:", gameState.tic)
    print("Game Variables:", gameState.game_variables, "\t Labels", gameState.labels)

    print("Screen Buffer:", gameState.screen_buffer.shape)
    processed = gameState.screen_buffer
    processed = preprocess_state_image(processed)
    if notebook:
        plt.imshow(processed, cmap='gray')
        plt.show()
    else:
        print(processed, "\n")


def preprocess_state_image(img):
    result = np.mean(img, axis=0)
    new_size_pil = (4 * result.shape[1]) // 5, (4 * result.shape[0]) // 5
    result = Image.fromarray(result)
    result = result.resize(new_size_pil, resample=Image.LANCZOS)
    result = np.array(result)
    print("Resizing from ", img.shape, " to ", result.shape)
    return result


def main_random(notebook=False):
    # Step 1: Initialize game enviornment

    game = DoomGame()
    game.load_config("vizdoom/scenarios/basic.cfg")
    game.init()

    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 5
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            print_game_state(state, notebook)
            img = state.screen_buffer
            misc = state.game_variables
            reward = game.make_action(random.choice(actions))
            print("\treward:", reward)
            time.sleep(0.01)
        print("Result:", game.get_total_reward())
        time.sleep(1)


def rational_trainer():
    # Step 1: Initialize game enviornment
    game = DoomGame()
    game.load_config("vizdoom/scenarios/basic.cfg")
    game.init()

    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    # Step 2: Intitialize replay memory capacity
    capacity = 10000 # HYPERPARAM
    memo = ReplayMemory(capacity)

    # Step 3: Construct and initialize policy network with random weights or weights from previous training sessions
    policy_nn = BasicDQN((192, 256, 1))

    # Step 3: Clone policy network to make target network
    target_nn = BasicDQN((192, 256, 1))
    target_nn.load_state_dict(policy_nn.state_dict())  # clones the weights of policy into target
    target_nn.eval()  # puts the target net into 'EVAL ONLY' mode, no gradients will be tracked or weights updated

    # Step 3b: Initialize an Optimizer
    learning_rate = 0.01  # HYPERPARAM
    optimizer = optim.Adam(params=policy_nn.parameters(), lr=learning_rate)

    # Step 4: Iterate over episodes
    episodes = 5
    explorer = Explorer(1, 0.05, 0.5)
    time_step_ctr = 0

    for i in range(episodes):

        # Step 5: Initialize game to a starting state
        game.new_episode()

        while not game.is_episode_finished():
            # Step 6: Select an action, either exploration or exploitation
            initial_state = game.get_state()

            if random.random() < explorer.curr_epsilon():  # Exploration
                action_todo = random.choice(actions)
            else:
                # TODO: Choose action via exploitation of neural net
                action_todo = random.choice(actions)  # REPLACE THIS LINE with action with the largest Q value

            # Step 7: Execute selected action in an emulator
            reward_received = game.make_action(action_todo)
            final_state = game.get_state()

            exp = Experience(initial_state, action_todo, final_state, reward_received)

            # Step 8: Store experience in replay memory
            memo.push(exp)

            # Step 9: Sample random batch from replay memory
            batch_size = 100
            batch = memo.sample(batch_size)

            # Step 10: Pre-process states from branch
            # TODO: just use the preprocess method

            # Step 11a: Pass states through the policy network and aquire output Q-values
            # TODO

            # Step 11b: Calculate target Q values. Pass successor states for each action through the target network
            #           use bellman equation to calculate target value
            # TODO

            # Step 11c: Calculate MSE (or any other) Loss between output and target values
            # TODO: use scipy loss functions, do not write ur own

            # Step 12: Use gradient descent, or ADAM, to update weights along the policy network
            # TODO: Dont write backprop urself, use the deep learning library (Keras or Pytorch)

            # Step 13: Every x timesteps, the weights of the target network are updated
            #          to be the weights of the policy network, small pertubations can be added
            time_step_lim = 20
            if time_step_ctr >= time_step_lim:
                # TODO: Update target network to copy of policy network
                time_step_ctr = 0

            # update required values
            time_step_ctr += 1
            time.sleep(0.01)
        print("Result:", game.get_total_reward())
        time.sleep(1)

    # Step 14: Save NN weights to a file so that it can later be read for testing the agent
    # TODO: use torch or keras builtin save method, do not write ur own or use pickle


if __name__ == '__main__':
    main_random(notebook=False)
