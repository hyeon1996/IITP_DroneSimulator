import argparse
from collections import namedtuple
from itertools import count

import os
import numpy as np


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter 

import wandb

class Actor(nn.Module):
    def __init__(self, state_dim, max_action, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, 2)  # Assuming 2-dimensional action space
        self.log_std_head = nn.Linear(256, 2)  # Assuming 2-dimensional action space
        self.max_action = max_action
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()

        self.state_dim_Q = state_dim
        self.action_dim_Q = action_dim
        self.fc1 = nn.Linear(self.state_dim_Q + self.action_dim_Q, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim_Q)
        a = a.reshape(-1, self.action_dim_Q)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QMIX(nn.Module):
    def __init__(self, num_agents):
        super(QMIX, self).__init__()
        self.num_agents = num_agents
        self.fc1 = nn.Linear(self.num_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, Q_values):
        x = F.relu(self.fc1(Q_values))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC_QMIX():
    def __init__(self, state_dim, action_dim, num_agents, Transition, learning_rate, capacity, gradient_steps,
                       batch_size, gamma, max_action, tau, device, agent_id):
        super(SAC_QMIX, self).__init__()

        ### wandb init ###
        wandb.init(project="sac_qmix", entity="sjrhee")
        ##################

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.capacity = capacity
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = device
        self.agent_id = agent_id

        self.policy_net = Actor(self.state_dim, self.max_action).to(self.device)
        self.value_net = Critic(self.state_dim).to(self.device)
        self.Q_nets = [Q(self.state_dim, self.action_dim).to(self.device) for _ in range(self.num_agents)]
        self.mixer = QMIX(self.num_agents).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.Q_optimizers = [optim.Adam(Q_net.parameters(), lr=self.learning_rate) for Q_net in self.Q_nets]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=self.learning_rate)

        self.replay_buffer = [self.Transition] * self.capacity
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()
        
        # #### Compare with Original Code ####
        # # Soft update unnecessary in QMIX, instead using Mixer network
        # for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
        #     target_param.data.copy_(param.data)
        # for i in range(num_agents):
        #    self.writer = SummaryWriter(f'/results/exp-QMIX/{i}')
        # ######################################

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()

    def store(self, s, a, r, s_, d):
        index = self.num_transition % self.capacity
        transition = self.Transition(s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def update(self):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))
        s = torch.tensor([t.s for t in self.replay_buffer]).float().to(self.device)
        a = torch.tensor([t.a for t in self.replay_buffer]).to(self.device)
        r = torch.tensor([t.r for t in self.replay_buffer]).to(self.device)
        s_ = torch.tensor([t.s_ for t in self.replay_buffer]).float().to(self.device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(self.device)

        for _ in range(self.gradient_steps):
            index = np.random.choice(range(self.capacity), self.batch_size, replace=False)
            bn_s = s[index]
            bn_a = a[index].reshape(-1, 1)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index]
            bn_d = d[index].reshape(-1, 1)

            target_value = self.value_net(bn_s_)
            next_q_values = [Q_net(bn_s_, bn_a) for Q_net in self.Q_nets]
            next_q_values_sum = torch.sum(torch.stack(next_q_values, dim=0), dim=0, keepdim=True)

            # Value loss
            value_loss = self.value_criterion(self.value)
             # Q loss
            q_values = [Q_net(bn_s, bn_a) for Q_net in self.Q_nets]
            q_loss = sum([self.Q_criterion(q, bn_r) for q in q_values])

            # QMIX loss
            mixer_input = torch.cat(next_q_values, dim=1)
            mix_output = self.mixer(mixer_input)
            target_mix = bn_r + (1 - bn_d) * self.gamma * target_value
            qmix_loss = self.Q_criterion(mix_output, target_mix.detach())

            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss = (-mix_output).mean()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            for Q_optimizer, q in zip(self.Q_optimizers, q_values):
                Q_optimizer.zero_grad()
                q_loss.backward()
                Q_optimizer.step()

            self.mixer_optimizer.zero_grad()
            qmix_loss.backward()
            self.mixer_optimizer.step()

            self.num_training += 1

            wandb.log({"Episode Reward" : episode_reward})
            


# #### Compare with Original Code ####
# def save(self):
#     save_path = f"/results/exp-QMIX/{self.agent_id}"
#     torch.save(self.policy_net.state_dict(), os.path.join(save_path, 'policy_net.pth'))
#     torch.save(self.value_net.state_dict(), os.path.join(save_path, 'value_net.pth'))
#     for i, Q_net in enumerate(self.Q_nets):
#         torch.save(Q_net.state_dict(), os.path.join(save_path, f'Q_net{i}.pth'))
#     print("====================================")
#     print(f"Model has been saved...---->agent{self.agent_id}")
#     print("====================================")

# def load(self):
#     load_path = f"/results/exp-QMIX/{self.agent_id}"
#     self.policy_net.load_state_dict(torch.load(os.path.join(load_path, 'policy_net.pth')))
#     self.value_net.load_state_dict(torch.load(os.path.join(load_path, 'value_net.pth')))
#     for i, Q_net in enumerate(self.Q_nets):
#         Q_net.load_state_dict(torch.load(os.path.join(load_path, f'Q_net{i}.pth')))
#     print(f"Model has been loaded ----> agent{self.agent_id}")
# ######################################