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


def print_game_state(gameState, notebook=False):
    if not gameState:
        print("\n\n ------------------------Terminal (None) State---------------------------\n\n")
        return
    print("Number:", gameState.number, "\t Tic:", gameState.tic)
    print("Game Variables:", gameState.game_variables, "\t Labels", gameState.labels)

    print("Screen Buffer:", gameState.screen_buffer.shape)
    processed = gameState.screen_buffer
    processed = preprocess_state_image(processed)
    if type(processed) == torch.Tensor:
        processed = processed.numpy()[0]
    if notebook:
        plt.imshow(processed, cmap='gray')
        plt.show()
    else:
        print(processed, "\n")


def preprocess_state_image(img):
    # TODO: Need to do more image preprocessing here, try to get the dimensions of the image down without loosing information
    return img


def get_action_dict(config_file_path):
    # TODO: Make an actual way to read the .cfg file
    #  create a dictionary map of the Move Names to their appropriate array representation
    left = torch.tensor([1, 0, 0]).to(device)
    right = torch.tensor([0, 1, 0]).to(device)
    shoot = torch.tensor([0, 0, 1]).to(device)

    actions = {'MOVE_LEFT': left, 'MOVE_RIGHT': right, 'ATTACK': shoot}
    return actions


def main_random(config_file_path, notebook=False):
    # Step 1: Initialize game enviornment

    game = DoomGame()
    game.load_config(config_file_path)
    game.init()
    left = torch.tensor([1, 0, 0]).to(device)
    right = torch.tensor([0, 1, 0]).to(device)
    shoot = torch.tensor([0, 0, 1]).to(device)

    actions = [left, right, shoot]

    episodes = 5
    for i in range(episodes):
        game.new_episode()
        state = game.get_state()
        print_game_state(state, notebook)
        while not game.is_episode_finished():
            # state = game.get_state()
            # print_game_state(state, notebook)
            action_todo = list(random.choice(actions))
            reward = game.make_action(action_todo)
            state = game.get_state()
            print_game_state(state, notebook)
            print("\treward:", reward)
            time.sleep(0.01)
        print("Result:", game.get_total_reward())
        time.sleep(1)
