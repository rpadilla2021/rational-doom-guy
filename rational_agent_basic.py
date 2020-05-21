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
    new_size_pil = (4*result.shape[1]) // 5, (4*result.shape[0]) // 5
    result = Image.fromarray(result)
    result = result.resize(new_size_pil, resample=Image.LANCZOS)
    result = np.array(result)
    print("Resizing from ", img.shape, " to ", result.shape)
    return result


def main(notebook=False):
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


if __name__ == '__main__':
    main(notebook=False)
