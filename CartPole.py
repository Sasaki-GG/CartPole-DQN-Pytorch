#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __author__ = "GG"

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.autograd import Variable

ENV_NAME = 'CartPole-v0'
EPISODE = 10000
STEP = 300
# Test episode
TEST_EPI = 10
# e-greed
EPSILON = 0.9
FINAL_EPSILON = 0.01
HIDDEN_DIM = 20
# Replay buffer size
REPLAY_SIZE = 10000
BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.9
# Update target net frequence
TARGET_UPDATE = 5
EPOCH = 10


class QNN(nn.Module):
    '''
    Single NN with two layers
    '''
    def __init__(self, input_dim, output_dim):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class DQN(nn.Module):
    '''
    DQN Agent. Policy Net (q_v) and Target Net (q_t).
    With replay buffer and e-greed.
    '''
    def __init__(self, env):
        super(DQN, self).__init__()
        # Replay buffer implemented with deque
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_DQN()

    def create_DQN(self):
        self.q_v = QNN(self.state_dim, self.action_dim)
        self.q_t = QNN(self.state_dim, self.action_dim)
        # Set eval() mode, as it's no need to update parameters in traning
        self.q_t.eval()
        self.q_t.load_state_dict(self.q_v.state_dict())

        self.optimizer = torch.optim.Adam(self.q_v.parameters(), lr=LR)
        # Huber Loss
        self.loss_function = nn.SmoothL1Loss()

    def train(self):
        '''
        Traing step
        '''
        # Sampling in replay buffer
        inputs = self.sample()
        # Batch of training data
        state_bat = [data[0] for data in inputs]
        action_bat = [data[1] for data in inputs]
        reward_bat = [data[2] for data in inputs]
        nxt_sta_bat = [data[3] for data in inputs if data is not None]
        # Useless
        over_bat = [1] * len(inputs)
        # Changed into tensor
        state_bat = torch.stack(state_bat).double().type('torch.FloatTensor')
        action_bat = torch.stack(action_bat).double().type('torch.FloatTensor')
        nxt_sta_bat = torch.stack(nxt_sta_bat).double().type('torch.FloatTensor')
        reward_bat = torch.tensor(reward_bat).double().type('torch.FloatTensor')
        over_bat = torch.tensor(over_bat).double().type('torch.FloatTensor')
        # Predict :  state estimating value (with actual action)
        y_pred = (self.q_v(state_bat) * action_bat).sum(1)
        # True value : GAMMA *  next state value (with max action) + reward
        y_true = GAMMA * torch.max(self.q_t(nxt_sta_bat).detach(),1)[0] + reward_bat

        loss = self.loss_function(y_true, y_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def greed_action(self, state):
        '''
        Policy Net prediction
        '''
        with torch.no_grad():
            out = self.q_v(state)
        return torch.argmax(out).numpy()

    def egreed_action(self, state):
        '''
        e-greed. Random influence decreases with steps going by.
        '''
        e = np.random.rand()
        if e > self.epsilon:
            return self.greed_action(state)
        else:
            return np.random.randint(0, self.action_dim)

        self.epsilon -= (EPSILON - FINAL_EPSILON) / 10000

    def perceive(self, state, action, reward, next_state, over):
        '''
        Restore replay buffer
        '''
        # Ont-hot action
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        one_hot_action = torch.tensor(one_hot_action)

        self.replay_buffer.append(
            (state, one_hot_action, reward, next_state, over))
        # Popleft of buffer when exceeding MAX_SIZE
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # Train when meeting batch size (much smaller than replay size)
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train()

    def sample(self):
        '''
        Smapling minibatch
        '''
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        return minibatch

    def copy_parameters(self):
        '''
        Update Target Net with Policy Net
        '''
        self.q_t.load_state_dict(self.q_v.state_dict())

def main():
    # Gym environment
    env = gym.make(ENV_NAME)
    # DQN agnet
    agent = DQN(env)

    for episode in range(EPISODE):
        # Initial
        state = env.reset()
        # Train
        for step in range(STEP):
            # To tensor
            state = torch.tensor(state).clone().detach()
            state = state.double().type('torch.FloatTensor')
            # e-greed
            action = agent.egreed_action(state)
            next_state, reward, over, _ = env.step(action)

            next_state = torch.tensor(next_state)
            next_state = next_state.double().type('torch.FloatTensor')
            # Reward
            if over:
                agent_reward = -1
            else:
                agent_reward = 1
            # Restore state
            agent.perceive(state, action, agent_reward, next_state, over)
            state = next_state
            if over:
                break
        # Update Tartget Net
        if episode % TARGET_UPDATE == 0:
            agent.copy_parameters()

        # Test
        if episode % 100 == 0:
            sum_reward = 0
            for i in range(TEST_EPI):
                state = env.reset()
                for j in range(STEP):
                    env.render()

                    state = torch.tensor(state)
                    state = state.double().type('torch.FloatTensor')

                    action = agent.greed_action(state)
                    state, reward, over, _ = env.step(action)

                    sum_reward += reward

                    if over:
                        break
            # Average reward
            ave = sum_reward / TEST_EPI
            print('EPI:{}\tReward:{}'.format(episode, ave))

if __name__ == "__main__":
    main()
