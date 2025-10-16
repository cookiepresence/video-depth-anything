import argparse
from pathlib import Path

def setup_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    # Student model parameters
    parser.add_argument('--student-model', type=str, choices=['vits', 'vitb', 'vitl',       'vitg'], default='vits',help='Student model size to use')
    parser.add_argument('--student-weights', type=str, default=None,
                      help='Path to pre-trained student model weights (optional)')
    parser.add_argument('--use-registers', action='store_true',
                      help='Use DinoV2 backbone with registers for student')

    # Teacher model parameters
    parser.add_argument('--teacher-model', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vitl',
                      help='Teacher model size to use')
    parser.add_argument('--teacher-weights', type=str, required=True,
                      help='Path to teacher (video depth) model weights')
    parser.add_argument('--teacher-backbone-weights', type=str, required=False,
                        help='Path to teacher (video depth) backbone weights (adapting from finetuned depth models)')

    # Dataset parameters
    parser.add_argument('--dataset-path', type=Path, required=True,
                      help='Path to the dataset root directory')
    parser.add_argument('--sport-name', type=str, default=None,
                      help='Optional sport name filter for dataset')

    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--train-batch-size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=8,
                      help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--backbone-lr', type=float, default=1e-6,
                      help='Learning rate for backbone parameters')
    parser.add_argument('--head-lr', type=float, default=1e-5,
                      help='Learning rate for DPT head parameters')
    parser.add_argument('--distill-lambda', type=float, default=0.5,
                      help='Weight for distillation loss vs depth loss (0-1)')

    # Logging and experiment parameters
    parser.add_argument('--use-wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name for the experiment run')
                      
    return parser