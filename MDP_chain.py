import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class MDP_chain():
    
    def __init__(self, n_states):
        self.start = 2
        self.n = n_states
        self.state = self.start
        
    def step(self, action):
        state_delta = action if action else - 1
        s_n = min(self.n, max(1, self.state + state_delta))
        r = 1 if (self.state == self.n and s_n == self.n) else 1/1000 if (self.state == 1 and s_n == 1) else 0
        self.state = s_n
        return torch.tensor([r], dtype=torch.float32, device=device), torch.tensor([self.state], dtype=torch.float32, device=device).unsqueeze(0)

    def reset(self):
        self.state = self.start
        return self.state
