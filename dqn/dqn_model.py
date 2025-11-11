import torch.nn as nn


# Uses ELU (ELU(x) = exp(x)-1 if x < 0 else x) instead of ReLU to preserve negative activations so Q(s,a) can
# represent "bad actions", avoiding dead neurons and improving stability.
class DQN(nn.Module):
    def __init__(self, input_dim=4*4*12, num_actions=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            # Keep informations for negative inputs
            nn.ELU(),        
            nn.Linear(128, 128),
            nn.ELU(),       
            nn.Linear(128, num_actions)  
        )

    def forward(self, x):
        return self.model(x)
