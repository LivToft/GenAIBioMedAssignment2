import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description="Test Evo2 Model Forward Pass")

    parser.add_argument("--run_dir", type=Path, help="Path to directory with experiment outputs", default=None)
    parser.add_argument("--n_samples", type=int, help="Number of representative samples to analyze", default=5)
    
    args = parser.parse_args()

    # if args.run_dir is None or not args.run_dir.exists():
    #     raise ValueError("Please provide a valid --run_dir path")

    # -----------------------------
    # Load predictions and targets
    # -----------------------------
    pred_file = args.run_dir / "pred.npy"
    target_file = args.run_dir / "target.npy"

    preds = np.load(pred_file)      # shape: [num_samples, H, W]
    targets = np.load(target_file)  # same shape

    assert preds.shape == targets.shape, "Predictions and targets must have same shape"

    num_samples = preds.shape[0]

    # -----------------------------
    # Compute Pearson correlation per sample
    # -----------------------------
    pccs = []

    for i in range(num_samples):
        pred_flat = preds[i].flatten()
        target_flat = targets[i].flatten()
        pcc, _ = pearsonr(pred_flat, target_flat)
        pccs.append(pcc)

    pccs = np.array(pccs)
    mean_pcc = np.mean(pccs)
    print(f"Average Pearson correlation across {num_samples} samples: {mean_pcc:.4f}")

    pcc_file = os.path.join(args.run_dir, "all_pccs.txt")

    with open(pcc_file, "w") as f:
        f.write(f"mean: {mean_pcc:.4f}\n")

        for idx, pcc in enumerate(pccs):
            f.write(f"sample_{idx}: {pcc:.4f}\n")

    print(f"PCCs saved to {pcc_file}")

    # -----------------------------
    # Select representative samples and plot
    # -----------------------------

    # Find sample with PCC closest to mean (or above)
    valid_indices = np.where(pccs >= mean_pcc)[0]
    diff = pccs[valid_indices] - mean_pcc

    if len(valid_indices) < args.n_samples:
        print(f"WARNING: Could not find {args.n_samples} samples with PCC > mean PCC. Using max.")
        n = len(valid_indices)
    else:
        n = args.n_samples

    rep_indices = valid_indices[np.argsort(diff)[:n]]

    for idx_rep in rep_indices:
        rep_pred = preds[idx_rep]
        rep_target = targets[idx_rep]
        rep_pcc = pccs[idx_rep]

        L = int(np.sqrt(rep_target.size))
        rep_pred = rep_pred.reshape(L, L)
        rep_target = rep_target.reshape(L, L)

        # Save figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im0 = axes[0].imshow(rep_target, cmap='viridis', origin='upper')
        axes[0].set_title("Ground Truth")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(rep_pred, cmap='viridis', origin='upper')
        axes[1].set_title("Prediction")
        plt.colorbar(im1, ax=axes[1])

        # Big overall title
        fig.suptitle(f"Sample {idx_rep}\n (PCC = {rep_pcc:.4f})", fontsize=14)
        fig.subplots_adjust(top=0.85)

        fig_name = os.path.join(args.run_dir, f"sample_{idx_rep}.png") 
        plt.savefig(fig_name, dpi=300) 
        plt.close(fig) 
        print(f"Saved figure for sample {idx_rep}")


if __name__ == "__main__":
    main()