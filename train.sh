#!/bin/bash
#SBATCH --partition=gpu_4_a100          # Partition name
#SBATCH --time=06:00:00                 # 
#SBATCH --gres=gpu:4                    # Job time limit (reduced from 48 hours)
#SBATCH --nodes=2                       # Number of nodes (reduced from 5)
#SBATCH --ntasks-per-node=32            # Tasks per node (reduced)
#SBATCH --cpus-per-task=2               # Number of threads per task
#SBATCH --mem=25600MB                  # Total memory (reduced from 510000MB)
#SBATCH --job-name=gradtts              # Job name
#SBATCH --output=output_%j.log          # Standard output log (%j is job ID)
#SBATCH --error=error_%j.log            # Standard error log (%j is job ID)

# Run your script
conda init
conda activate gradtts_env
export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
python train.py 
