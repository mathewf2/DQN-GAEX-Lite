import torch
import torch.nn as nn

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class Generator(nn.Module):
    def __init__(self, n_states, lr=0.001):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(50, n_states),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.model(x)
        return x

    def optimize(self, criterion, x):
        self.zero_grad()
        label = torch.ones_like(x, device=device)
        loss = criterion(x, label)
        loss.backward()
        self.optimizer.step()

class Discriminator(nn.Module):
    def __init__(self, n_states=10, lr=0.001):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        ).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def step(self, x_real, x_fake):
        self.zero_grad()

        pred_real = self(x_real)
        pred_fake = self(x_fake.detach())

        return pred_real, pred_fake

    def optimize(self, criterion, pred_real, pred_fake):
        batch_size = pred_real.size()[0]
        label_real = torch.ones((batch_size,1), device=device)
        label_fake = torch.zeros((batch_size,1), device=device)
        
        loss = criterion(pred_real, label_real)
        loss.backward()

        loss = criterion(pred_fake, label_fake)
        loss.backward()

        self.optimizer.step()

class GAN():
    def __init__(self, n_states):
        self.G = Generator(n_states)
        self.D = Discriminator(n_states)

    def step(self, x_real):
        noise = torch.randn_like(x_real, device=device)
        x_fake = self.G(noise)
        pred_real, pred_fake = self.D.step(x_real, x_fake)
        return pred_real, pred_fake, x_fake

    def optimize(self, criterion, x_fake, pred_real, pred_fake):
        self.D.optimize(criterion, pred_real, pred_fake)
        self.G.optimize(criterion, self.D(x_fake))
