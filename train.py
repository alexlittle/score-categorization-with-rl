

import time
import os
import argparse
import importlib

from tensorboardX import SummaryWriter


from dqn_model.dqnagent import DQNAgent


DQN_CONFIG = {'hidden_dims': 128, 'num_episodes': 10, 'batch_size': 64, 'epsilon_fn': 'linear',
              'epsilon_lin_start': 1.0, 'epsilon_lin_end': 0.01, 'replaybuffer_capacity': 500000, 'gamma': 0.99,
              'lr': 1e-4, 'tau': 0.005, 'data_file': 'course_bbb_2013b.csv'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', type=str, help="Simulator library to use", required=True)
    parser.add_argument('--data_file',
                        type=str,
                        help="Which CSV file to use",
                        default=DQN_CONFIG['data_file'])
    parser.add_argument('--num_episodes',
                        type=int,
                        help="Number of Episodes to run",
                        default=DQN_CONFIG['num_episodes'])
    args = parser.parse_args()

    # Update config based on any command line params provided
    DQN_CONFIG['num_episodes'] = args.num_episodes
    DQN_CONFIG['epsilon_decay'] = DQN_CONFIG['num_episodes'] * 3 / 4
    DQN_CONFIG['simulator'] = args.simulator
    DQN_CONFIG['data_file'] = args.data_file

    # get data file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(dir_path, 'data', DQN_CONFIG['data_file'])

    try:
        simulator = importlib.import_module(args.simulator)
    except ModuleNotFoundError:
        print(f"Error: The simulator '{args.simulator}' could not be found.")
        return

    # init Simulator Environment
    env = simulator.LearningPredictorEnv(data_file_path)



    start_time = time.time()

    os.makedirs("./runs/", exist_ok=True)
    writer = SummaryWriter(log_dir="./runs/")

    agent = DQNAgent(env, DQN_CONFIG, writer)
    agent.train()

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

if __name__ == "__main__":
    main()





