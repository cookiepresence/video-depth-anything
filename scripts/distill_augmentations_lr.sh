#!/bin/bash
#SBATCH --job-name=depth-distill
#SBATCH --output=logs/.depth_distill_%j.out
#SBATCH --error=logs/.depth_distill_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Sync dependencies (if using uv)
uv sync

# Activate conda environment (adjust to your environment name)
source .venv/bin/activate
# OR: conda activate depth_env

# Run the distillation script
python distillation.py \
    --student-model vits \
    --student-weights  ../depth-anything-v2/saved_models/best_model_depth_v2_augmentation_2421854.pth \
    --teacher-model vits \
    --teacher-weights ./checkpoints/video_depth_anything_vits.pth \
    --teacher-backbone-weights ../depth-anything-v2/saved_models/best_model_depth_v2_2421365.pth \
    --dataset-path /ssd_scratch/soccernet/ \
    --sport-name basketball \
    --seed 42 \
    --train-batch-size 1 \
    --val-batch-size 4 \
    --epochs 5 \
    --backbone-lr 1e-6 \
    --head-lr 1e-4 \
    --distill-lambda 1.0 \
    --use-wandb \
    --experiment-name "depth_distill_augmentations_lr_${SLURM_JOB_ID}"

# Print end time
echo "End time: $(date)"
