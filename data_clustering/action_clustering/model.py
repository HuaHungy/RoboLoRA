"""Small temporal MLP + state/context encoders, trained without skill labels."""

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class ActionRepresentation(nn.Module):
    def __init__(self, length, action_dim, state_dim, visual_dim, hidden_dim=256,
                 latent_dim=128, visual_weight=0.0):
        super().__init__()
        self.length, self.action_dim = length, action_dim
        self.visual_weight = visual_weight
        # Flattening preserves temporal order; this is a fixed-window temporal MLP.
        self.action_encoder = mlp(length * action_dim, hidden_dim, hidden_dim)
        self.state_encoder = mlp(state_dim, hidden_dim, hidden_dim)
        self.visual_encoder = mlp(visual_dim, hidden_dim, hidden_dim) if visual_dim else None
        self.context_dropout = nn.Dropout(0.5)
        self.projector = mlp(hidden_dim * (3 if visual_dim else 2), hidden_dim, latent_dim)
        self.action_decoder = mlp(latent_dim, hidden_dim, length * action_dim)
        self.state_decoder = mlp(latent_dim, hidden_dim, state_dim)

    def encode(self, action, state, visual):
        parts = [F.normalize(self.action_encoder(action.flatten(1)), dim=-1),
                 F.normalize(self.state_encoder(state), dim=-1)]
        if self.visual_encoder is not None:
            parts.append(self.visual_weight * self.context_dropout(
                F.normalize(self.visual_encoder(visual), dim=-1)))
        return F.normalize(self.projector(torch.cat(parts, dim=-1)), dim=-1)

    def forward(self, action, state, visual):
        z = self.encode(action, state, visual)
        return z, self.action_decoder(z).reshape(-1, self.length, self.action_dim), self.state_decoder(z)
