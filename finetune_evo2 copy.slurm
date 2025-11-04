#!/bin/bash
#SBATCH -A cis250160p
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/finetune/%A_%a.out      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=slurm_outputs/finetune/%A_%a.out       # Standard error file (%j will be replaced with job ID)

# load modules
module load anaconda3/2024.10-1
module load cuda/12.4.0

# activate environment
conda activate /ocean/projects/cis250160p/ltoft/evo2

# finetune model
cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" finetune_evo2_cmds.txt`
eval $cmd
exit