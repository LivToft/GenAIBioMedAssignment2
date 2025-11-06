import torch
import torch.nn as nn
import torch.nn.functional as F

class PairHead(nn.Module):
    def __init__(self, input_size, proj_dim, pair_hidden_dim, conv_channels):
        super().__init__()

        self.pair_hidden_dim = pair_hidden_dim

        # project each embedding down to lower dimension
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # learn pairwise interactions
        self.pair_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, pair_hidden_dim),
            nn.GELU(),
            nn.Linear(pair_hidden_dim, pair_hidden_dim),
            nn.GELU()
        )

        # conv refinement for learning 2D spatial structure
        self.refine = nn.Sequential(
            nn.Conv2d(pair_hidden_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(conv_channels, 1, kernel_size=1)
        )

    def forward(self, h):  
        B, L, D = h.shape   # (B, L, D)

        # Project embeddings
        p = self.proj(h)  # (B, L, proj_dim)

        # Expand to pairwise dimensions
        pi = p.unsqueeze(2).expand(B, L, L, -1)  # (B, L, L, proj_dim)
        pj = p.unsqueeze(1).expand(B, L, L, -1)  # (B, L, L, proj_dim)

        # Create pair features
        pair_feat = torch.cat([pi + pj, (pi - pj).abs(), pi * pj], dim=-1)  # (B, L, L, 4*proj_dim)
        pair_feat = pair_feat.view(B, L*L, -1)

        # Pass through MLP
        pair_out = self.pair_mlp(pair_feat)  # (B, L*L, pair_hidden_dim)
        pair_out = pair_out.view(B, L, L, self.pair_hidden_dim).permute(0, 3, 1, 2)  # (B, pair_hidden_dim, L, L)

        # Conv refinement
        contact_map = self.refine(pair_out)  # (B, 1, L, L)

        # Collapse to a single scalar score per token embedding pair
        contact_map_flat = contact_map.view(B, L*L) # (B, L*L)

        return contact_map_flat
