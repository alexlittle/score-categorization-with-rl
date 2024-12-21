import os
import time
import torch
import torch.nn as nn
import random
from itertools import count
import torch.optim as optim


from rnn_dqn_model.rnn_dqn import LSTM_DQN
from rnn_dqn_model.replaymemory import ReplayBuffer

class RNNDQNAgent:
    def __init__(self, simulator, agent_config, writer):

        self.config = agent_config
        self.writer = writer
        self.simulator = simulator
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.n_actions = self.simulator.action_space.n
        self.n_observations = len(self.simulator.reset())
        self.policy_net = LSTM_DQN(self.n_observations, self.config['hidden_dims'], self.n_actions)
        self.target_net = LSTM_DQN(self.n_observations, self.config['hidden_dims'], self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['lr'])
        self.replay_buffer = ReplayBuffer(self.config['replaybuffer_capacity'])
        self.steps_done = 0
        self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else
                        "cpu"
                    )
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.epsilon = self.config['epsilon_lin_start']
        self.episode_durations = []
        self.episode_rewards = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.policy_net.output_size))
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values, _ = self.policy_net(state)
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            return torch.argmax(q_values, dim=1).item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare the batch
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

        # if state is just [0.5] then unsqueeze(-1) if [0.5, 0.6...] then unsqueeze(1)
        states = states.unsqueeze(1)

        hidden_state = self.policy_net.init_hidden(self.batch_size, self.device)

        # Get current Q-values from policy network
        q_values, next_hidden_state = self.policy_net(states, hidden_state)
        q_values = q_values.squeeze(1)

        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_states = next_states.unsqueeze(1)

        next_q_values, _ = self.target_net(next_states, hidden_state)
        next_q_values = next_q_values.squeeze(1)
        next_state_values = next_q_values.max(1)[0]

        # Expected Q-values
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return next_hidden_state

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):

        for episode in range(self.config['num_episodes']):
            state = self.simulator.reset()
            ep_start_time = time.time()
            total_reward = 0

            for t in count():
                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.simulator.step(action)
                self.replay_buffer.push(state, action, next_state, reward, done)

                self.optimize_model()

                state = next_state
                total_reward += reward

                if episode % 10 == 0:
                    self.update_target_network()

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(total_reward)
                    break

            # Decay epsilon
            self.epsilon = max(self.config['epsilon_lin_end'],
                          self.epsilon - (self.config['epsilon_lin_start'] - self.config['epsilon_lin_end']) / self.config['epsilon_decay'])
            ep_time = time.time() - ep_start_time
            self.writer.add_scalar('Episode Reward', total_reward, episode)
            self.writer.add_scalar('Episode Duration', self.episode_durations[-1], episode)
            self.writer.add_scalar('Episode Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Episode Time', ep_time, episode)

            print(f"Episode {episode + 1}/{self.config['num_episodes']}, Duration {self.episode_durations[-1]}, "
                  f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, Time: {ep_time:.2f}")

