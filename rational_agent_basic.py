from vizdoom import *
import random, time
from pprint import pprint
import numpy as np
from PIL import Image
from DataStructures import *
from DQN import BasicDQN
import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
import matplotlib.pyplot as plt


def print_game_state(gameState, notebook=False):
    if not gameState:
        print("\n\n ------------------------Terminal (None) State---------------------------\n\n")
        return
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
    #TODO: Need to do more image preprocessing here, try to get the dimensions of the image down without loosing information
    result = np.mean(img, axis=0)
    result = Image.fromarray(result)
    # Do PIL Pre Proccessing here
    result = np.array(result)
    #Shrinking vertically
    result = result[70:185]
    #Shrinking horizontally
    result = result[:,75:250]
    print("Original size ", img.shape, " to ", result.shape)
    return result


def main_random(notebook=False):
    # Step 1: Initialize game enviornment

    game = DoomGame()
    game.load_config("vizdoom/scenarios/basic.cfg")
    game.init()

    left = torch.tensor([1, 0, 0])
    right = torch.tensor([0, 1, 0])
    shoot = torch.tensor([0, 0, 1])

    actions = [left, right, shoot]

    episodes = 5
    for i in range(episodes):
        game.new_episode()
        state = game.get_state()
        print_game_state(state, notebook)
        while not game.is_episode_finished():
            # state = game.get_state()
            #print_game_state(state, notebook)
            action_todo = list(random.choice(actions))
            reward = game.make_action(action_todo)
            state = game.get_state()
            print_game_state(state, notebook)
            print("\treward:", reward)
            time.sleep(0.01)
        print("Result:", game.get_total_reward())
        time.sleep(1)


def rational_trainer(notebook=False):
    # Step 1: Initialize game enviornment
    game = DoomGame()
    game.load_config("vizdoom/scenarios/basic.cfg")
    game.init()

    left = torch.tensor([1, 0, 0])
    right = torch.tensor([0, 1, 0])
    shoot = torch.tensor([0, 0, 1])

    actions = [left, right, shoot]

    # Step 2: Intitialize replay memory capacity
    capacity = 10000  # HYPERPARAM
    memo = ReplayMemory(capacity)

    # Step 3: Construct and initialize policy network with random weights or weights from previous training sessions
    game.new_episode()
    test_state = game.get_state()
    processed_test = preprocess_state_image(test_state.screen_buffer)
    policy_nn = BasicDQN(processed_test.shape, actions)

    # Step 3: Clone policy network to make target network
    target_nn = BasicDQN(processed_test.shape, actions)
    target_nn.load_state_dict(policy_nn.state_dict())  # clones the weights of policy into target
    target_nn.eval()  # puts the target net into 'EVAL ONLY' mode, no gradients will be tracked or weights updated

    # Step 3b: Initialize an Optimizer
    learning_rate = 0.05  # HYPERPARAM
    optimizer = optim.Adam(params=policy_nn.parameters(), lr=learning_rate)

    # Step 4: Iterate over episodes
    episodes = 250
    explorer = Explorer(1, 0.05, 0.0001)
    time_step_ctr = 0

    for i in range(episodes):

        # Step 5: Initialize game to a starting state
        game.new_episode()

        while not game.is_episode_finished():
            # Step 6: Select an action, either exploration or exploitation
            initial_state = game.get_state()
            processed_s = preprocess_state_image(initial_state.screen_buffer)


            skip_rate = 3 # Hyperparam
            if random.random() < explorer.curr_epsilon():  # Exploration
                action_todo = random.choice(actions)
                #print("Random Action:", action_todo)
            else:
                action_todo = policy_nn.select_best_action(processed_s)  # exploitation
                #print("Optimal Action:", action_todo)

            # Step 7: Execute selected action in an emulator
            action_todo = list(action_todo)
            reward_received = game.make_action(action_todo, skip_rate)
            final_state = game.get_state()

            # Step 8 Preprocess and create expeience states
            if final_state == None: # We are in  a terminal state
                processed_s_prime = np.zeros_like(processed_test)
            else:
                processed_s_prime = preprocess_state_image(final_state.screen_buffer)

            exp = Experience(torch.from_numpy(processed_s).unsqueeze(0), torch.tensor([action_todo]), torch.from_numpy(processed_s_prime).unsqueeze(0), torch.tensor([reward_received]))

            # Step 9: Store experience in replay memory
            memo.push(exp)

            # Step 10: Sample random batch from replay memory
            batch_size = 200
            loss = torch.tensor(-1)
            if memo.can_sample(batch_size):
                states, actions, next_states, rewards = memo.sample_tensors(batch_size)

                # Step 11a: Pass states through the policy network and aquire output Q-values
                pred_q_vals = DQN.get_current_QVals(policy_nn, states, actions)

                # Step 11b: Calculate target Q values. Pass successor states for each action through the target network
                #           use bellman equation to calculate target value
                gamma_discount = 0.9 # HYPERPARAM
                next_q_vals = DQN.get_next_QVals(target_nn, next_states)
                target_q_vals = rewards + gamma_discount*next_q_vals

                # Step 11c: Calculate MSE (or any other) Loss between output and target values
                loss = F.mse_loss(pred_q_vals, target_q_vals) # replace the nones
                #print("LOSS: ", loss.item())

                # Step 12: Use gradient descent, or ADAM, to update weights along the policy network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            # Step 13: Every x timesteps, the weights of the target network are updated
            #          to be the weights of the policy network, small pertubations can be added
            target_update_steps = 200
            if time_step_ctr == target_update_steps:
                print("Updating Target Net")
                target_nn.load_state_dict(policy_nn.state_dict())
                time_step_ctr = 0

            # update required values
            time_step_ctr += 1
            time.sleep(0.001)
        print("Episode", i)
        print("Explorer", explorer.curr_epsilon())
        print("Result:", game.get_total_reward())
        print("Last LOSS:", loss.item())
        time.sleep(0.01)

    # Step 14: Save NN weights to a file so that it can later be read for testing the agent
    torch.save(policy_nn.state_dict(), 'rational_net_basic.model')


def rational_tester(model_path, notebook=False):
    # Initialize game enviornment
    game = DoomGame()
    game.load_config("vizdoom/scenarios/basic.cfg")
    game.init()

    left = torch.tensor([1, 0, 0])
    right = torch.tensor([0, 1, 0])
    shoot = torch.tensor([0, 0, 1])

    actions = [left, right, shoot]

    # Loading the policy net from model path
    game.new_episode()
    test_state = game.get_state()
    processed_test = preprocess_state_image(test_state.screen_buffer)
    policy_nn = BasicDQN(processed_test.shape, actions)

    policy_nn.load_state_dict(torch.load(model_path))
    policy_nn.eval()

    episodes = 5
    for i in range(episodes):
        game.new_episode()
        state = game.get_state()
        print_game_state(state, notebook)
        while not game.is_episode_finished():
            initial_state = game.get_state()
            processed_s = preprocess_state_image(initial_state.screen_buffer)

            action_todo = policy_nn.select_best_action(processed_s, 4)
            action_todo = list(action_todo)
            reward = game.make_action(action_todo)

            state = game.get_state()
            print_game_state(state, notebook)

            print("\treward:", reward)

            time.sleep(0.02)

        print("Result:", game.get_total_reward())
        time.sleep(1)


if __name__ == '__main__':
    # rational_trainer()
    # rational_tester('rational_net_basic.model')
    main_random(notebook=False) # Change this to true to see what the preproccessed images look like