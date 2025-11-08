import torch
import torch.nn as nn
import torch.nn.functional as F

class PairHead(nn.Module): 
    def __init__(self, input_size, proj_dim, pair_hidden_dim, conv_channels): 
        super().__init__() 
        self.pair_hidden_dim = pair_hidden_dim # project each embedding down to lower dimension 
        
        self.proj = nn.Sequential( 
            nn.Linear(input_size, proj_dim), 
            nn.LayerNorm(proj_dim), 
            nn.GELU() ) 
        
        # learn pairwise interactions 
        self.pair_mlp = nn.Sequential( 
            nn.Linear(proj_dim * 3, pair_hidden_dim), 
            nn.GELU(), 
            nn.Linear(pair_hidden_dim, pair_hidden_dim), 
            nn.GELU() ) 
        
        # conv refinement for learning 2D spatial structure 
        self.refine = nn.Sequential( 
            nn.Conv2d(pair_hidden_dim, conv_channels, kernel_size=3, padding=1), 
            nn.GELU(), 
            nn.Conv2d(conv_channels, 1, kernel_size=1) ) 
        
    def forward(self, h): 
        B, L, D = h.shape # (B, L, D) 

        # Project embeddings 
        p = self.proj(h) # (B, L, proj_dim) 
        
        # Expand to pairwise dimensions 
        pi = p.unsqueeze(2).expand(B, L, L, -1) # (B, L, L, proj_dim) 
        pj = p.unsqueeze(1).expand(B, L, L, -1) # (B, L, L, proj_dim) 
        
        # Create pair features 
        pair_feat = torch.cat([pi + pj, (pi - pj).abs(), pi * pj], dim=-1) # (B, L, L, 3*proj_dim) 
        pair_feat = pair_feat.view(B, L*L, -1) # (B, L*L, 3*proj_dim)
        
        # Pass through MLP 
        pair_out = self.pair_mlp(pair_feat) # (B, L*L, pair_hidden_dim) 
        pair_out = pair_out.view(B, L, L, self.pair_hidden_dim).permute(0, 3, 1, 2) # (B, pair_hidden_dim, L, L) 
        
        # Conv refinement 
        contact_map = self.refine(pair_out) # (B, 1, L, L) 
        
        # Collapse to a single scalar score per token embedding pair 
        contact_map_flat = contact_map.view(B, L*L) # (B, L*L) 
        
        return contact_map_flat


class PairHead2(nn.Module):
    def __init__(self, input_size, proj_dim, pair_hidden_dim, conv_channels):
        super().__init__()

        self.pair_hidden_dim = pair_hidden_dim
        self.proj_dim = proj_dim

        # project each embedding down to lower dimension
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        self.dist_emb = nn.Embedding(512, proj_dim)

        # learn pairwise interactions
        self.pair_cnn = nn.Sequential(
            nn.Conv2d(2*proj_dim, pair_hidden_dim, kernel_size=1), 
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

        # Create pair features as difference and product of embeddings
        diff = torch.abs(p[:, :, None, :] - p[:, None, :, :])
        prod = p[:, :, None, :] * p[:, None, :, :] 
        pair_feat = torch.cat([diff, prod], dim=-1)     # (1, L, L, 2*proj_dim)
        pair_feat = F.layer_norm(pair_feat, pair_feat.shape[-1:])   # Normalize per pair to prevent magnitude bias

        # Embed pairwise distances
        pos = torch.arange(L, device=h.device)
        dist = torch.abs(pos[None, :, None] - pos[None, None, :])  # (1, L, L)
        dist = dist.clamp(max=self.dist_emb.num_embeddings - 1)
        dist_emb = self.dist_emb(dist)  # (1, L, L, proj_dim)
        dist_emb = torch.cat([dist_emb, dist_emb], dim=-1)  # (1, L, L, 2*proj_dim)

        # Combine dist embeddings wuth pairwise features
        pair_feat = pair_feat + dist_emb  # (B, L, L, 2*proj_dim) 
        pair_feat = pair_feat.permute(0, 3, 1, 2)   # (B, C, L, L)
        
        # Pass through CNNs
        pair_out = self.pair_cnn(pair_feat) # (B, pair_hidden_dim, L, L)
        contact_map = self.refine_cnn(pair_out).squeeze(1)  # (B, L, L)

        # Enforce symmetry
        contact_map = 0.5 * (contact_map + contact_map.transpose(1, 2))

        # Flatten
        contact_map_flat = contact_map.view(B, L * L)

        return contact_map_flat
