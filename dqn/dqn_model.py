import torch.nn as nn


class residual_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),  #nn.ELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)

# Uses ELU (ELU(x) = exp(x)-1 if x < 0 else x) instead of ReLU to preserve negative activations so Q(s,a) can
# represent "bad actions", avoiding dead neurons and improving stability.
class DQN(nn.Module):
    def __init__(self, input_dim=4*4*12, num_actions=4, num_res_blocks=3, hidden_dim=64):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.model = nn.ModuleList([residual_block(hidden_dim) for _ in range(num_res_blocks)])
        self.Advantage = nn.Linear(hidden_dim, num_actions)
        self.Value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        for i in range(len(self.model)):
            x = self.model[i](x)
        features = x
        advantage = self.Advantage(features)
        value = self.Value(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))