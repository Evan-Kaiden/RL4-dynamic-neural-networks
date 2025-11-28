import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CAgent(nn.Module):
    def __init__(self, context_dim : int, hidden : int, full_action_space : int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, full_action_space)
        )
        self.critic = nn.Linear(hidden, 1)

        self.full_action_space = full_action_space

    def forward(self, x : torch.Tensor, possible_actions : int):
        """Returns masked actions"""
        device = x.device
        B = x.size(0)
        enc = self.encoder(x)
        act = self.actor(enc)
        val = self.critic(enc)
        allowed_space = torch.full((B, possible_actions), 0.0, dtype=act.dtype, device=device)
        masked_space  = torch.full((B, self.full_action_space - possible_actions), -1e9, dtype=act.dtype, device=device)
        mask = torch.cat([allowed_space, masked_space], dim=1)
        act = act + mask
        return act, val