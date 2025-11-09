'''
Code adapted from https://github.com/ma-compbio/DNALONGBENCH
'''

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ruamel.yaml as yaml
import os

# from prediction_heads.pairhead import *

from evo2 import Evo2

from datasets.akita_dataset import get_dataloader

# TODO: WANDB INTEGRATION - Import the library
import wandb

data_path = "/ocean/projects/cis250160p/rhettiar/contact_map_prediction/extra_data.1m/tfrecords"

def main():
    """
    Test sequence prediction accuracy using Evo2 models.
    Expected results for forward pass:
    - Evo 2 40B 1m: Loss ~0.216, Accuracy ~91.67%
    - Evo 2 7B 1m: Loss ~0.348, Accuracy ~86.35%
    - Evo 2 1B base: Loss ~0.502, Accuracy ~79.56%
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Forward Pass")

    parser.add_argument("--config", type=str, help="Path to config file", default="configs/baseline.yaml")
    parser.add_argument("--output_dir", type=str, default="evo2/contact_map", help="Output folder")

    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'], default='evo2_7b', help="Model to test")

    # ---------------- WandB ----------------
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity key")
    parser.add_argument("--run_name", type=str, default='i_forgot_to_name', help="run name")

    # ---------------- Parameters ----------------
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--pred_head_arch", type=str, default=None, help="Prediction head architecture")
    parser.add_argument("--proj_dim", type=int, default=None, help="Projection dimension")
    parser.add_argument("--pair_hidden_dim", type=int, default=None, help="Hidden layer size in pair MLP")
    parser.add_argument("--conv_channels", type=int, default=None, help="Number of out channels refinement CNN layers")
    parser.add_argument("--downweight_diag", type=int, default=False, help="Down-weight diagonals in loss")


    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    args = parser.parse_args()

    # setup output folder
    os.makedirs(args.output_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    

    # TODO: WANDB INTEGRATION - Initialize a new run
    wandb.login(key=args.wandb_key)

    run = wandb.init(
            name    = args.run_name,
            reinit  = True, 
            entity  = args.wandb_entity,
            project = args.wandb_project,
            config  = vars(args)
    )
    
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # Initialize model
    model = Evo2(args.model_name)

    train_loader = get_dataloader(f"{data_path}/train-*.tfr", "HFF")
    valid_loader = get_dataloader(f"{data_path}/valid-*.tfr", "HFF")

    if args.pred_head_arch == "baseline":
        task_layer = nn.Linear(4096*2, 1).to("cuda")
    elif args.pred_head_arch == "pairhead":
        task_layer = PairHead(input_size=4096, proj_dim=args.proj_dim, pair_hidden_dim=args.pair_hidden_dim, conv_channels=args.conv_channels).to("cuda")
    else:
        raise NotImplementedError("Only baseline and pair prediction head architectures have been implemented.")
    
    optimizer = torch.optim.Adam(task_layer.parameters(), lr=6e-4)
    
    if args.epochs <= 10:
        step_size = 3
        gamma = 0.1
    else:
        step_size = args.epochs//6
        gamma = 0.4

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    val_loss = 10000
    dic = {0: "A", 1: "C", 2: "G", 3: "T"}
    
    for epoch in range(1, args.epochs):
        task_layer.train() # Set model to training mode
        
        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} | Training", unit="batch")

        for batch in train_pbar:
            # zero the parameter gradients
            optimizer.zero_grad()
            
            seq, scores = batch
            seq_string = [dic[ind.item()] for ind in seq[0]]
            seq_string = "".join(seq_string)
            
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq_string),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            scores = scores.to("cuda:0")
            
            layer_name = 'blocks.28.mlp.l3'
            outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
            hiddens = embeddings[layer_name]
            hiddens = torch.mean(hiddens.reshape(hiddens.size(0), -1, 2048, hiddens.size(-1)), dim=2)

            if args.pred_head_arch == 'baseline':
                norm = torch.sqrt(torch.sum(hiddens * hiddens, dim=-1)).unsqueeze(-1) # [B, L]
                norm = torch.bmm(norm, norm.transpose(1, 2))
                outs = (torch.bmm(hiddens, hiddens.transpose(1, 2))/norm).reshape(hiddens.size(0), -1)
                matrix = hiddens[0]
                vec1 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1).transpose(0, 1)
                vec2 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1)
                vec3 = torch.cat((vec2, vec1), dim=-1).reshape(-1, hiddens.size(-1)*2)
                outs = task_layer(vec3.float()).unsqueeze(0).squeeze(-1)
                loss = F.mse_loss(outs, scores)

            elif args.pred_head_arch == 'pairhead':
                
                outs = task_layer(hiddens.float())
                
                if args.downweight_diag:
                    B, L, _ = hiddens.shape
                    mask = torch.ones((B, L * L), device=outs.device)
                    diag_idx = torch.arange(L, device=outs.device) * (L + 1)
                    mask[:, diag_idx] = 0.1
                    loss = F.mse_loss(outs*mask, scores*mask)
                else:
                    loss = F.mse_loss(outs, scores)
            
            loss.backward()
            optimizer.step()

            # TODO: WANDB INTEGRATION - Log step-wise training loss
            wandb.log({
                'train_loss': loss,
            })

            # Update the progress bar with the current loss
            train_pbar.set_postfix(loss=f"{loss.cpu().item():.4f}")

        lr_scheduler.step()

        task_layer.eval() # Set model to evaluation mode
        with torch.no_grad():
            this_val_loss = []
            
            # Wrap valid_loader with tqdm for a progress bar
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch} | Validation", unit="batch")
            for batch in valid_pbar:
                
                seq, scores = batch
                seq_string = [dic[ind.item()] for ind in seq[0]]
                seq_string = "".join(seq_string)
                
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq_string),
                    dtype=torch.int,
                ).unsqueeze(0).to('cuda:0')
                scores = scores.to("cuda:0")
                
                layer_name = 'blocks.28.mlp.l3'
                outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
                hiddens = embeddings[layer_name]  # [1, 102400, 4096]
                hiddens = torch.mean(hiddens.reshape(hiddens.size(0), -1, 2048, hiddens.size(-1)), dim=2)  # [B, 50, dim]

                if args.pred_head_arch == 'baseline':
                    norm = torch.sqrt(torch.sum(hiddens * hiddens, dim=-1)).unsqueeze(-1) # [B, L]
                    norm = torch.bmm(norm, norm.transpose(1, 2))
                    outs = (torch.bmm(hiddens, hiddens.transpose(1, 2))/norm).reshape(hiddens.size(0), -1)
                    matrix = hiddens[0]
                    vec1 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1).transpose(0, 1)
                    vec2 = matrix.view(-1, 1, hiddens.size(-1)).repeat(1, hiddens.size(1), 1)
                    vec3 = torch.cat((vec2, vec1), dim=-1).reshape(-1, hiddens.size(-1)*2)
                    outs = task_layer(vec3.float()).unsqueeze(0).squeeze(-1) #[1, 50*50, 1]

                elif args.pred_head_arch == 'pairhead':
                    outs = task_layer(hiddens.float())
                
                loss = F.mse_loss(outs, scores)
                this_val_loss.append(loss.cpu().item())
                
                # Update the progress bar with the current validation loss
                valid_pbar.set_postfix(loss=f"{loss.cpu().item():.4f}")

            this_val_loss_average = np.average(this_val_loss)
            current_lr = lr_scheduler.get_last_lr()[0]
            
            # TODO: WANDB INTEGRATION - Log epoch-wise validation loss and learning rate (make sure to log the epoch number as well)
            wandb.log({
                'val_loss': this_val_loss_average,
                'lr': current_lr,
                'epoch_num': epoch,
            })

            if this_val_loss_average < val_loss:
                val_loss = this_val_loss_average
                torch.save(task_layer, os.path.join(save_dir, 'model.pt'))

                # TODO: WANDB INTEGRATION - Log the best validation loss so far
                wandb.log({
                    'best_val_loss': val_loss,
                })

            # Print the final validation loss for the epoch
            print(f"Epoch {epoch} final validation loss: {val_loss:.4f}")

    # save config file
    experiment_config = vars(args)
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        file_yaml.dump(experiment_config, f)

    # TODO: WANDB INTEGRATION - Finish the run
    run.finish()

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
    
if __name__ == "__main__":
    main()