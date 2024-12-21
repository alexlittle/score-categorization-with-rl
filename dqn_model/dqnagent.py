'''
DQNAgent

Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''


import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import count
from dqn_model.dqn import DQN
from dqn_model.replaymemory import ReplayMemory, Transition


class DQNAgent():

    def __init__(self, simulator, dqn_agent_config, writer):
        self.simulator = simulator
        self.dqn_config = dqn_agent_config
        self.writer = writer
        self.memory = ReplayMemory(self.dqn_config['replaybuffer_capacity'])
        self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else
                        "cpu"
                    )
        self.n_actions = self.simulator.action_space.n
        self.n_observations = len(self.simulator.reset())
        self.policy_net = DQN(self.n_observations, self.dqn_config['hidden_dims'], self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.dqn_config['hidden_dims'], self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.dqn_config['lr'], amsgrad=True)
        self.epsilon = self.dqn_config['epsilon_lin_start']
        self.episode_durations = []
        self.episode_rewards = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[self.simulator.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.dqn_config['gamma']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        for episode in range(self.dqn_config['num_episodes']):
            # Initialize the environment and get its state
            state = self.simulator.reset()
            ep_start_time = time.time()
            total_reward = 0
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, done, _  = self.simulator.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)


                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                #if len(observation) == self.n_observations:
                #    print("added")
                self.memory.push(state, action, next_state, reward)
                #else:
                #print("not added")

                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(self.dqn_config['batch_size'])

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] \
                                                 * self.dqn_config['tau'] + target_net_state_dict[key] * (
                                                 1 - self.dqn_config['tau'])
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    print(state)
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(total_reward)
                    break

            ep_time = time.time() - ep_start_time
            self.writer.add_scalar('Episode Reward', total_reward, episode)
            self.writer.add_scalar('Episode Duration', self.episode_durations[-1], episode)
            self.writer.add_scalar('Episode Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Episode Time', ep_time, episode)

            print(f"Episode {episode+1}/{self.dqn_config['num_episodes']}, Duration {self.episode_durations[-1]}, "
                f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, Time: {ep_time:.2f}")
            self.epsilon = max(self.dqn_config['epsilon_lin_end'],
                          self.epsilon - (self.dqn_config['epsilon_lin_start']
                                     - self.dqn_config['epsilon_lin_end']) / self.dqn_config['epsilon_decay'])
