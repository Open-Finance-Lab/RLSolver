# Non-Autoregressive TSP Solver

A PyTorch implementation of a neural network for solving the Traveling Salesman Problem.

## Requirements

```bash
pip install torch torchvision torchaudio
pip install tqdm numpy

# Optional: for heuristic comparison
pip install elkai
```

## Usage

### Training

Train the model (uses all available GPUs automatically):

```bash
python train.py
```

The trained model will be saved to `model.pth`.

### Inference

Evaluate the trained model:

```bash
# Basic evaluation
python inference.py --checkpoint model.pth

# With greedy decoding
python inference.py --checkpoint model.pth --greedy

# With multiple samples
python inference.py --checkpoint model.pth --num_samples 10
```

## Command Line Options

### Inference Options

```bash
python inference.py [OPTIONS]

Options:
  --checkpoint PATH     Model checkpoint path (default: model.pth)
  --num_nodes INT      Number of TSP nodes (default: 30)
  --num_test INT       Number of test instances (default: 2000)
  --batch_size INT     Batch size for evaluation (default: 256)
  --num_samples INT    Samples per instance (default: 1)
  --greedy            Use greedy decoding
  --no_heuristic      Skip heuristic comparison
  --seed INT          Random seed (default: 111)
  --device DEVICE     Device to use (default: cuda)
```

## Files

- `train.py` - Training script
- `inference.py` - Evaluation script
- `config.py` - Configuration parameters
- `model.pth` - Saved model (created after training)
