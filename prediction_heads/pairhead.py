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
        self.pair_cnn = nn.Sequential(
            nn.Conv2d(proj_dim*proj_dim, pair_hidden_dim, kernel_size=1),  # 1x1 conv to process features per pair
            nn.GELU(),
            nn.Conv2d(pair_hidden_dim, pair_hidden_dim, kernel_size=1),
            nn.GELU()
        )

        # conv refinement for learning 2D spatial structure
        self.refine_cnn = nn.Sequential(
            nn.Conv2d(pair_hidden_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(conv_channels, 1, kernel_size=1)
        )

    def forward(self, h):  
        B, L, D = h.shape   # (B, L, D)

        # Project embeddings
        p = self.proj(h)  # (B, L, proj_dim)

        # Create pair features as outer product 
        pair_feat = torch.einsum('bid,bjd->bijd', p, p) # (B,L,L,D,D)
        pair_feat = pair_feat.view(B, L, L, D*D).permute(0, 3, 1, 2)  # (B, D*D, L, L)

        # Pass through CNN for pair representation learning
        pair_out = self.pair_cnn(pair_feat)   # (B, pair_hidden_dim, L, L)

        # CNN refinement
        contact_map = self.refine_cnn(pair_out)  # (B, 1, L, L)
        contact_map = contact_map.squeeze(1)     # (B, L, L)

        # Enforce symmetry
        contact_map = 0.5 * (contact_map + contact_map.transpose(1, 2))

        # Flatten for scoring
        contact_map_flat = contact_map.view(B, L*L)  # (B, L*L)

        return contact_map_flat
