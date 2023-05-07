import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from GAN import Discriminator, GAN

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class DQN_net(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.model(x)

class DQN():
    def __init__(self, dqn_update_freq, n_observations=1, n_actions=2, batch_size=300, lr=0.005, gamma=0.99, eps_strt=1.0, eps_end=0.0, eps_decay=0.0005, trgt_upd_freq=50):
        self.pi = DQN_net(n_observations, n_actions).to(device)
        self.trgt = DQN_net(n_observations, n_actions).to(device)
        self.trgt.load_state_dict(self.pi.state_dict())
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_strt = eps_strt
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.trgt_upd_freq = trgt_upd_freq
        self.steps = 0
        self.updates = 0
        self.dqn_upd_freq = dqn_update_freq

        self.optimizer = optim.Adam(self.pi.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(self.batch_size)
        self.sns = torch.Tensor()
        self.sn_mask = torch.Tensor()
        
    def remember(self, *args):
        self.memory.push(*args)

    def sample(self):
        return self.memory.sample(self.batch_size)

    def action(self, s):
        epsilon = self.eps_end + (self.eps_strt - self.eps_end) * math.exp(-self.steps * self.eps_decay)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                return self.pi(s).max(1)[1].view(1,1).to(device)
        else:
            return torch.tensor([[random.choice([0,1])]], device=device, dtype=torch.long)

    def step(self):
        if len(self.memory) < self.batch_size:
            return None, None

        transitions = self.sample()
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        sa_vals = self.pi(state_batch).gather(1, action_batch)
        
        sn_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        sns = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        self.sns = sns
        self.sn_mask = sn_mask

        nxt_sa_vals = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            nxt_sa_vals[sn_mask] = self.trgt(sns).max(1)[0]

        exp_sa_vals = reward_batch + self.gamma * nxt_sa_vals
        
        return sa_vals, exp_sa_vals

    def optimize(self, sa_vals, exp_sa_vals):
        if not self.steps % self.dqn_upd_freq:
            loss = self.criterion(sa_vals, exp_sa_vals.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.updates += 1

        if not self.updates % self.trgt_upd_freq:
            self.trgt.load_state_dict(self.pi.state_dict())

        self.steps += 1


class DQN_D(DQN):
    def __init__(self, n_states, dqn_update_freq, discrm_update_freq, n_observations=1, n_actions=2, beta=1.0):
        super().__init__(dqn_update_freq, n_observations, n_actions)
        self.D = Discriminator(n_states)
        self.beta = beta
        self.discrm_criterion = nn.BCELoss()
        self.discrm_upd_freq = discrm_update_freq

    def optimize(self, sa_vals, exp_sa_vals):
        x_real = F.one_hot(self.sns.cpu().squeeze(1).long()-1, num_classes=10).float().to(device)
        x_fake = torch.randn(x_real.size(), device=device)

        pred_real, pred_fake = self.D.step(x_real, x_fake)

        r_intr = (self.beta * (1 - pred_real.detach())**2).squeeze(1).to(device)
        exp_sa_vals[self.sn_mask.detach()] += r_intr

        super().optimize(sa_vals, exp_sa_vals)

        if not self.steps % self.discrm_upd_freq:
            self.D.optimize(self.discrm_criterion, pred_real, pred_fake)

class DQN_GAEX(DQN):
    def __init__(self, n_states, dqn_update_freq, gan_upd_freq, n_observations=1, n_actions=2, beta=1.0):
        super().__init__(dqn_update_freq, n_observations, n_actions)
        self.GAN = GAN(n_states)
        self.beta = beta
        self.gan_criterion = nn.BCELoss()
        self.gan_upd_freq = gan_upd_freq

    def optimize(self, sa_vals, exp_sa_vals):
        x_real = F.one_hot(self.sns.cpu().squeeze(1).long()-1, num_classes=10).float().to(device)
        pred_real, pred_fake, x_fake = self.GAN.step(x_real)

        r_intr = (self.beta * (1 - pred_real.detach())**2).squeeze(1).to(device)
        exp_sa_vals[self.sn_mask.detach()] += r_intr
        
        super().optimize(sa_vals, exp_sa_vals)

        if not self.steps % self.gan_upd_freq:
            self.GAN.optimize(self.gan_criterion, x_fake, pred_real, pred_fake)
