

import time
import os
import torch
import argparse
import importlib
import json

from datetime import datetime
from tensorboardX import SummaryWriter

from rnn_dqn_model.rnndqnagent import RNNDQNAgent
from gyms import helper

RNNDQN_CONFIG = {'hidden_dims': 128,
              'num_episodes': 10,
              'batch_size': 64,
              'epsilon_fn': 'linear',
              'epsilon_lin_start': 1.0,
              'epsilon_lin_end': 0.01,
              'replaybuffer_capacity': 500000,
              'gamma': 0.99,
              'lr': 1e-4,
              'tau': 0.005,
              'data_file': 'course_bbb_2013b-train.csv',
              'categories_range_start': 50,
              'categories_range_end': 91,
              'grade_boundaries': 10 }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator',
                        type=str,
                        help="Simulator library to use",
                        required=True)
    parser.add_argument('--hidden_dims',
                        type=int,
                        help="Size of hidden dimensions",
                        default=RNNDQN_CONFIG['hidden_dims'])
    parser.add_argument('--data_file',
                        type=str,
                        help="Which CSV file to use",
                        default=RNNDQN_CONFIG['data_file'])
    parser.add_argument('--num_episodes',
                        type=int,
                        help="Number of Episodes to run",
                        default=RNNDQN_CONFIG['num_episodes'])
    parser.add_argument('--batch_size',
                        type=int,
                        help="Batch size",
                        default=RNNDQN_CONFIG['batch_size'])
    parser.add_argument('--epsilon_lin_start',
                        type=int,
                        help="Starting value of epsilon (for linear decay)",
                        default=RNNDQN_CONFIG['epsilon_lin_start'])
    parser.add_argument('--epsilon_lin_end',
                        type=int,
                        help="Ending value of epsilon (for linear decay)",
                        default=RNNDQN_CONFIG['epsilon_lin_end'])
    parser.add_argument('--grade_boundaries',
                        type=int,
                        help="Grade boundary step, eg, with 15, boundaries will be 50,65,80",
                        default=RNNDQN_CONFIG['grade_boundaries'])
    parser.add_argument('--categories_range_start',
                        type=int,
                        help="Start of score category range",
                        default=RNNDQN_CONFIG['categories_range_start'])
    parser.add_argument('--categories_range_end',
                        type=int,
                        help="End of score category range",
                        default=RNNDQN_CONFIG['categories_range_end'])
    args = parser.parse_args()

    # Update config based on any command line params provided
    RNNDQN_CONFIG['simulator'] = args.simulator
    RNNDQN_CONFIG['hidden_dims'] = args.hidden_dims
    RNNDQN_CONFIG['data_file'] = args.data_file
    RNNDQN_CONFIG['num_episodes'] = args.num_episodes
    RNNDQN_CONFIG['batch_size'] = args.batch_size
    RNNDQN_CONFIG['epsilon_lin_start'] = args.epsilon_lin_start
    RNNDQN_CONFIG['epsilon_lin_end'] = args.epsilon_lin_end
    RNNDQN_CONFIG['grade_boundaries'] = args.grade_boundaries
    RNNDQN_CONFIG['categories_range_start'] = args.categories_range_start
    RNNDQN_CONFIG['categories_range_end'] = args.categories_range_end

    RNNDQN_CONFIG['num_categories'] = helper.num_score_categories(RNNDQN_CONFIG['categories_range_start'],
                                                               RNNDQN_CONFIG['categories_range_end'],
                                                               RNNDQN_CONFIG['grade_boundaries'] )

    RNNDQN_CONFIG['epsilon_decay'] = RNNDQN_CONFIG['num_episodes'] * 3 / 4



    # get data file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(dir_path, 'data', RNNDQN_CONFIG['data_file'])

    try:
        simulator = importlib.import_module(args.simulator)
    except ModuleNotFoundError:
        print(f"Error: The simulator '{args.simulator}' could not be found.")
        return

    # init Simulator Environment
    env = simulator.LearningPredictorEnv(data_file_path, RNNDQN_CONFIG)
    RNNDQN_CONFIG['max_sequence_length'] = env.max_sequence_length
    start_time = time.time()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

    os.makedirs(f"./runs/{timestamp}/", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./runs/{timestamp}/")

    agent = RNNDQNAgent(env, RNNDQN_CONFIG, writer)
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

    RNNDQN_CONFIG['average_reward'] = average_reward
    RNNDQN_CONFIG['average_reward_last_half'] = average_reward_last_half
    RNNDQN_CONFIG['average_reward_last_100'] = average_reward_last_100
    RNNDQN_CONFIG['elapsed_time'] = elapsed_time
    ############################################
    # save model state dict and config used
    ############################################
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_output_file = os.path.join(dir_path, 'output', timestamp +".pth")
    torch.save(agent.policy_net.state_dict(), model_output_file)

    config_output_file = os.path.join(dir_path, 'output', timestamp + ".json")
    with open(config_output_file, "w") as file:
        json.dump(RNNDQN_CONFIG, file, indent=4)

if __name__ == "__main__":
    main()





