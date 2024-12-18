

import time
import os
import torch
import argparse
import importlib
import json

from datetime import datetime
from tensorboardX import SummaryWriter

from dqn_model.dqnagent import DQNAgent


DQN_CONFIG = {'hidden_dims': 128,
              'num_episodes': 10,
              'batch_size': 64,
              'epsilon_fn': 'linear',
              'epsilon_lin_start': 1.0,
              'epsilon_lin_end': 0.01,
              'replaybuffer_capacity': 500000,
              'gamma': 0.99,
              'lr': 1e-4,
              'tau': 0.005,
              'data_file': 'course_bbb_2013b.csv',
              'grade_boundaries': 10,
              'max_sequence_length': 11,
              'num_categories': 6}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator',
                        type=str,
                        help="Simulator library to use",
                        required=True)
    parser.add_argument('--hidden_dims',
                        type=int,
                        help="Size of hidden dimensions",
                        default=DQN_CONFIG['hidden_dims'])
    parser.add_argument('--data_file',
                        type=str,
                        help="Which CSV file to use",
                        default=DQN_CONFIG['data_file'])
    parser.add_argument('--num_episodes',
                        type=int,
                        help="Number of Episodes to run",
                        default=DQN_CONFIG['num_episodes'])
    parser.add_argument('--batch_size',
                        type=int,
                        help="Batch size",
                        default=DQN_CONFIG['batch_size'])
    parser.add_argument('--epsilon_lin_start',
                        type=int,
                        help="Starting value of epsilon (for linear decay)",
                        default=DQN_CONFIG['epsilon_lin_start'])
    parser.add_argument('--epsilon_lin_end',
                        type=int,
                        help="Ending value of epsilon (for linear decay)",
                        default=DQN_CONFIG['epsilon_lin_end'])
    parser.add_argument('--grade_boundaries',
                        type=int,
                        help="Grade boundary step, eg, with 15, boundaries will be 50,65,80",
                        default=DQN_CONFIG['grade_boundaries'])
    parser.add_argument('--max_sequence_length',
                        type=int,
                        help="Max length of observation sequence",
                        default=DQN_CONFIG['max_sequence_length'])
    parser.add_argument('--num_categories',
                        type=int,
                        help="number of score categories",
                        default=DQN_CONFIG['num_categories'])
    args = parser.parse_args()

    # Update config based on any command line params provided
    DQN_CONFIG['simulator'] = args.simulator
    DQN_CONFIG['hidden_dims'] = args.hidden_dims
    DQN_CONFIG['data_file'] = args.data_file
    DQN_CONFIG['num_episodes'] = args.num_episodes
    DQN_CONFIG['batch_size'] = args.batch_size
    DQN_CONFIG['epsilon_lin_start'] = args.epsilon_lin_start
    DQN_CONFIG['epsilon_lin_end'] = args.epsilon_lin_end
    DQN_CONFIG['grade_boundaries'] = args.grade_boundaries
    DQN_CONFIG['max_sequence_length'] = args.max_sequence_length
    DQN_CONFIG['num_categories'] = args.num_categories

    DQN_CONFIG['epsilon_decay'] = DQN_CONFIG['num_episodes'] * 3 / 4



    # get data file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(dir_path, 'data', DQN_CONFIG['data_file'])

    try:
        simulator = importlib.import_module(args.simulator)
    except ModuleNotFoundError:
        print(f"Error: The simulator '{args.simulator}' could not be found.")
        return

    # init Simulator Environment
    env = simulator.LearningPredictorEnv(data_file_path,
                                         max_sequence_length=DQN_CONFIG['max_sequence_length'],
                                         grade_boundaries=DQN_CONFIG['grade_boundaries'],
                                         num_categories=DQN_CONFIG['num_categories'])

    start_time = time.time()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

    os.makedirs(f"./runs/{timestamp}/", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./runs/{timestamp}/")

    agent = DQNAgent(env, DQN_CONFIG, writer)
    agent.train()

    ############################################
    # completed training - output results
    ############################################
    print('Complete')
    writer.close()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # average of all episode rewards
    average_reward = sum(agent.episode_rewards) / len(agent.episode_rewards)
    print(f"Average rewards (overall): {average_reward:.2f} ")

    # average of last half of episode rewards
    last_half = agent.episode_rewards[len(agent.episode_rewards) // 2:]
    average_reward_last_half = sum(last_half) / len(last_half)
    print(f"Average rewards (last half): {average_reward_last_half:.2f} ")

    last_100 = agent.episode_rewards[len(agent.episode_rewards) - 100:]
    average_reward_last_100 = sum(last_100) / len(last_100)
    print(f"Average rewards (last 100): {average_reward_last_100:.2f} ")

    print(f"Total runtime: {elapsed_time:.2f} seconds")

    DQN_CONFIG['average_reward'] = average_reward
    DQN_CONFIG['average_reward_last_half'] = average_reward_last_half
    DQN_CONFIG['average_reward_last_100'] = average_reward_last_100
    DQN_CONFIG['elapsed_time'] = elapsed_time
    ############################################
    # save model state dict and config used
    ############################################
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_output_file = os.path.join(dir_path, 'model_state_dicts', timestamp +".pth")
    torch.save(agent.policy_net.state_dict(), model_output_file)

    config_output_file = os.path.join(dir_path, 'model_state_dicts', timestamp + ".json")
    with open(config_output_file, "w") as file:
        json.dump(DQN_CONFIG, file, indent=4)

if __name__ == "__main__":
    main()





