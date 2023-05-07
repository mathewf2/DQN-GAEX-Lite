from DQN import DQN, DQN_D, DQN_GAEX
from MDP_chain import MDP_chain
from tqdm import tqdm
import torch
import torch.nn as nn
import math
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

n_states = 10
steps = n_states + 9
n_episodes = 1000
dqn_update_freq = 2 * n_episodes
gan_update_freq = 2 * n_episodes
env = MDP_chain(n_states)

dqn = DQN()

rewards = []
for episode in tqdm(range(n_episodes)):
    s = env.reset()
    s = torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0)
    reward = 0
    for t in range(steps):
        a = dqn.action(s)
        r, s_n = env.step(a)

        if t+1 == steps:
            s_n = None
        else:
            reward += r.cpu().item()

        dqn.remember(s, a, s_n, r)
        dqn.optimize()

        s = s_n

    rewards.append(reward)

plt.plot(rewards)
