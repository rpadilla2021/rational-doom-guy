from vizdoom import *
import random, time
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    episodes = 10
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
    # TODO

    # Step 3: Construct and initialize policy network with random weights or weights from previous training sessions
    # TODO

    # Step 3: Clone policy network to make target network
    # TODO

    # Step 4: Iterate over episodes
    episodes = 5
    epsilon = 0.05
    time_step_ctr = 0

    for i in range(episodes):

        # Step 5: Initialize game to a starting state
        game.new_episode()

        while not game.is_episode_finished():
            # Step 6: Select an action, either exploration or exploitation
            initial_state = game.get_state()

            if random.random() < epsilon:  # Exploration
                action_todo = random.choice(actions)
            else:
                # TODO: Choose action via exploitation of neural net
                action_todo = random.choice(actions)  # REPLACE THIS LINE with action with the largest Q value

            # Step 7: Execute selected action in an emulator
            reward_received = game.make_action(action_todo)
            final_state = game.get_state()

            experience = initial_state, action_todo, reward_received, final_state

            # Step 8: Store experience in replay memory
            # TODO

            # Step 9: Sample random batch from replay memory
            # TODO

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
