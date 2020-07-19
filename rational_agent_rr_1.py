from vizdoom import *
import random, time
from DataStructures import *
from DQN import *
import DQN
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

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
        processed = processed.cpu().numpy()[0]
    if notebook:
        plt.imshow(processed, cmap='gray')
        plt.show()
    else:
        print(processed, "\n")


def preprocess_state_image(img):
    result = torch.tensor(img).to(device).float()

    result = torch.mean(result, dim=0)

    result = result[75:200, 75:200].unsqueeze(0).unsqueeze(0).to(device)

    result = F.interpolate(result, scale_factor=(0.9, 0.5), mode='bilinear', recompute_scale_factor=True,
                           align_corners=True).squeeze(0)
    return result


def main_random(notebook=False):
    # Step 1: Initialize game enviornment

    game = DoomGame()
    game.load_config("vizdoom/scenarios/health_gathering.cfg")
    game.init()
    left = torch.tensor([1, 0, 0]).to(device)
    right = torch.tensor([0, 1, 0]).to(device)
    straight = torch.tensor([0, 0, 1]).to(device)

    actions = [left, right, straight]

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


def rational_trainer(notebook=False):
    # Step 1: Initialize game enviornment
    game = DoomGame()
    game.load_config("vizdoom/scenarios/health_gathering.cfg")
    game.init()

    left = torch.tensor([1, 0, 0]).to(device)
    right = torch.tensor([0, 1, 0]).to(device)
    straight = torch.tensor([0, 0, 1]).to(device)

    actions = [left, right, straight]

    # Step 2: Intitialize replay memory capacity
    capacity = 10000  # HYPERPARAM
    memo = ReplayMemory(capacity)

    # Step 3: Construct and initialize policy network with random weights or weights from previous training sessions
    game.new_episode()
    test_state = game.get_state()
    processed_test = preprocess_state_image(test_state.screen_buffer)
    policy_nn = HealthDQN(processed_test.shape, actions).to(device)

    # Step 3: Clone policy network to make target network
    target_nn = HealthDQN(processed_test.shape, actions).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())  # clones the weights of policy into target
    target_nn.eval()  # puts the target net into 'EVAL ONLY' mode, no gradients will be tracked or weights updated

    # Step 3b: Initialize an Optimizer
    learning_rate = 0.25  # HYPERPARAM
    optimizer = optim.Adam(params=policy_nn.parameters(), lr=learning_rate)
    # Step 4: Iterate over episodes
    episodes = 2000
    explorer = Explorer(1, 0.05, 0.000009)
    time_step_ctr = 0

    rawards = []

    for i in range(episodes):

        # Step 5: Initialize game to a starting state
        game.new_episode()

        while not game.is_episode_finished():
            # Step 6: Select an action, either exploration or exploitation
            initial_state = game.get_state()
            processed_s = preprocess_state_image(initial_state.screen_buffer)

            if notebook:
                print_game_state(initial_state, notebook)

            skip_rate = 4  # Hyperparam
            if random.random() < explorer.curr_epsilon():  # Exploration
                action_todo = random.choice(actions)
                # print("Random Action:", action_todo)
            else:
                action_todo = policy_nn.select_best_action(processed_s)  # exploitation
                # print("Optimal Action:", action_todo)

            # Step 7: Execute selected action in an emulator
            action_todo = list(action_todo)
            reward_received = game.make_action(action_todo, skip_rate)
            final_state = game.get_state()

            # Step 8 Preprocess and create expeience states
            if final_state == None:  # We are in  a terminal state
                processed_s_prime = torch.zeros_like(processed_test).to(device)
            else:
                processed_s_prime = preprocess_state_image(final_state.screen_buffer)

            exp = Experience(processed_s,
                             torch.tensor([action_todo]).to(device),
                             processed_s_prime,
                             torch.tensor([reward_received]).to(device))

            # Step 9: Store experience in replay memory
            memo.push(exp)

            # Step 10: Sample random batch from replay memory
            batch_size = 100
            loss = torch.tensor(-1).to(device)
            if memo.can_sample(batch_size):
                states, actions, next_states, rewards = memo.sample_tensors(batch_size)

                # Step 11a: Pass states through the policy network and aquire output Q-values
                pred_q_vals = DQN.get_current_QVals(policy_nn, states, actions)

                # Step 11b: Calculate target Q values. Pass successor states for each action through the target network
                #           use bellman equation to calculate target value
                gamma_discount = 0.999  # HYPERPARAM
                next_q_vals = DQN.get_next_QVals(target_nn, next_states)
                target_q_vals = rewards + gamma_discount * next_q_vals

                # Step 11c: Calculate MSE (or any other) Loss between output and target values
                loss = F.mse_loss(pred_q_vals, target_q_vals)  # replace the nones
                # print("LOSS: ", loss.item())

                # Step 12: Use gradient descent, or ADAM, to update weights along the policy network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step 13: Every x timesteps, the weights of the target network are updated
            #          to be the weights of the policy network, small pertubations can be added
            target_update_steps = 2000
            if time_step_ctr == target_update_steps:
                print("Updating Target Net-------------------------------------------------------------------------")
                target_nn.load_state_dict(policy_nn.state_dict())
                time_step_ctr = 0

            # update required values
            time_step_ctr += 1
            # time.sleep(0.001)
        print("Episode", i)
        print("Explorer", explorer.curr_epsilon())
        final_reward = game.get_total_reward()
        print("Result:", final_reward)
        print("Last LOSS:", loss.item())
        rawards.append(final_reward)
        if i == episodes - 1:
            plot(rawards, 100, True)
        else:
            plot(rawards, 100)
        time.sleep(0.005)
        torch.cuda.empty_cache()

    # Step 14: Save NN weights to a file so that it can later be read for testing the agent
    torch.save(policy_nn.state_dict(), 'rational_net_basic.model')


def rational_tester(model_path, notebook=False):
    # Initialize game enviornment
    game = DoomGame()
    game.load_config("vizdoom/scenarios/deadly_corridor.cfg")
    game.init()
    print(device)
    left = torch.tensor([1, 0, 0]).to(device)
    right = torch.tensor([0, 1, 0]).to(device)
    straight = torch.tensor([0, 0, 1]).to(device)
    actions = [left, right, straight]
    # Loading the policy net from model path
    game.new_episode()
    test_state = game.get_state()
    processed_test = preprocess_state_image(test_state.screen_buffer)
    policy_nn = HealthDQN(processed_test.shape, actions).to(device)

    policy_nn.load_state_dict(torch.load(model_path))
    policy_nn.eval()

    episodes = 20
    for i in range(episodes):
        game.new_episode()
        state = game.get_state()
        if notebook:
            print_game_state(state, notebook)
        while not game.is_episode_finished():
            initial_state = game.get_state()
            processed_s = preprocess_state_image(initial_state.screen_buffer)

            action_todo = policy_nn.select_best_action(processed_s, show=True)
            action_todo = list(action_todo)
            reward = game.make_action(action_todo, 4)

            state = game.get_state()
            print_game_state(state, notebook)

            print("\treward:", reward)

            time.sleep(0.05)

        print("Result:", game.get_total_reward())
        time.sleep(1)


if __name__ == '__main__':
    rational_trainer(notebook=False)
    # main_random(notebook=True)
