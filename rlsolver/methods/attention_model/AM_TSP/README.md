# TSP Solver

A deep reinforcement learning approach for solving the Traveling Salesman Problem (TSP) using Policy Optimization with Multiple Optima (POMO) algorithm.

## Features

- **POMO Algorithm**: Generates multiple diverse solutions starting from different nodes
- **Attention Mechanism**: Uses transformer-style attention for node selection
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Optimized Performance**: Includes torch.compile and zero-copy optimizations
- **Heuristic Comparison**: Evaluates against classical TSP solvers (elkai/LKH)

## Requirements

```bash
pip install torch torchvision tqdm numpy
pip install elkai  # Optional: for heuristic baseline comparison
```

## Quick Start

### Training

```bash
# Single GPU training
python train.py

# The script automatically detects available GPUs and uses distributed training
```

### Configuration

Edit `config.py` to adjust training parameters:

```python
NUM_NODES = 20          # TSP problem size (number of cities)
NUM_EPOCHS = 10       # Training epochs
BATCH_SIZE = 256      # Batch size
LR = 0.0002          # Learning rate
```

### Key Training Parameters

- `NUM_NODES`: Number of cities in TSP instance (default: 20)
- `NUM_TRAIN_ENVS`: Number of POMO rollouts per instance (default: same as SEQ_LEN)
- `EMBEDDING_SIZE`: Neural network embedding dimension
- `NUM_TR_DATASET`: Number of training instances
- `NUM_TE_DATASET`: Number of test instances

### Model Files

After training:
- `model.pth`: Best trained model
- `checkpoints/`: Detailed checkpoints and training logs
- `results/`: Inference results (if applicable)

## File Structure

```
├── config.py          # Configuration parameters
├── train.py           # Main training script
├── trainer.py         # POMO training logic
├── models.py          # Neural network models
├── layers.py          # Attention layers
├── env.py             # TSP environment
├── dataset.py         # Data loading
└── util.py           # Utility functions
```

## How It Works

1. **Data Generation**: Random TSP instances with nodes in [0,1]×[0,1]
2. **POMO Training**: Multiple rollouts starting from different nodes
3. **Attention Model**: Encoder-decoder architecture with cross-attention
4. **Shared Baseline**: Uses mean of all POMO rollouts as baseline
5. **Loss Function**: Policy gradient with advantage estimation

## Training Output

The training will show:
```
Epoch 0: Train Loss: X.XXXX, Train Reward: X.XXXX, Eval Reward: X.XXXX
Gap vs Heuristic: X.XXXx
New best model! Model saved...
```

## Performance

- **Gap**: Ratio compared to heuristic solver (lower is better)
- **Reward**: Negative tour length (higher is better)
- **POMO Size**: More rollouts generally improve solution quality

## GPU Memory

For larger problems or batch sizes, you may need to:
- Reduce `BATCH_SIZE`
- Reduce `NUM_TRAIN_ENVS` (POMO size)
- Use gradient accumulation

## Advanced Usage

### Custom Problem Size
```python
# In config.py
NUM_NODES = 50  # For 50-city TSP
NUM_TRAIN_ENVS = 50  # POMO rollouts
```

### Evaluation Only
```python
# Load trained model for inference
model = TSPActor(embedding_size, hidden_size, seq_len, n_head, C)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## Notes

- Training time scales with problem size and number of epochs
- POMO generates diverse solutions by starting from different nodes
- Best results typically achieved with POMO size equal to problem size
- Heuristic comparison requires `elkai` package installation

