# Max-Cut PPO Training and Evaluation

## Training

Single GPU Training:
```bash
python train_ddp.py
```

Multi-GPU Training:
```bash
python launch.py
```

After training, `model.pth` will be generated.

## Parameter Configuration

Modify parameters in `config.py`:

```python
# Main Parameters
epochs = 1000          # Number of training epochs
batch_size = 8192      # Batch size
lr = 2e-4             # Learning rate
num_parallel_envs = 8  # Number of parallel environments

# Environment Parameters
episode_length_multiplier = 2  # Multiplier for maximum steps
tabu_tenure = 10              # Tabu tenure length
```

## Evaluation

Place `evaluate.py` in the `rlsolver/methods/maxcut/` directory:

```bash
cd rlsolver/methods/maxcut/
python evaluate.py
```

The evaluation script will:
- Load `model.pth`
- Read test graphs from `../../data`
- Save results to `../../result`