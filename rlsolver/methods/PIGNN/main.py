"""
PIGNN Main Entry Point - Train/Inference Dispatcher
Based on ECO S2V architecture pattern

This file serves as the unified entry point that dispatches to appropriate
train or inference scripts based on configuration settings.
"""

import sys
import os
from rlsolver.methods.PIGNN.config import (
    TRAIN_INFERENCE, PROBLEM, Problem, MODEL_CHECKPOINT_DIR, RESULTS_DIR
)


def print_configuration():
    """Print current configuration for debugging."""
    print("="*60)
    print("PIGNN (Physics-Inspired Graph Neural Network)")
    print("="*60)
    print(f"Mode: {'TRAINING' if TRAIN_INFERENCE == 0 else 'INFERENCE'}")
    print(f"Problem: {PROBLEM.name}")
    print("="*60)


def dispatch_train_mode():
    """Dispatch to appropriate training script based on problem type."""
    if PROBLEM == Problem.maxcut:
        from rlsolver.methods.PIGNN.train.train_maxcut import run
        print("Loading MaxCut training module...")
    elif PROBLEM == Problem.MIS:
        from rlsolver.methods.PIGNN.train.train_MIS import run
        print("Loading MIS training module...")
    elif PROBLEM == Problem.graph_coloring:
        from rlsolver.methods.PIGNN.train.train_graph_coloring import run
        print("Loading Graph Coloring training module...")
    else:
        raise ValueError(f"Unsupported problem type: {PROBLEM}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

    # Run the training
    run()


def dispatch_inference_mode():
    """Dispatch to appropriate inference script based on problem type."""
    if PROBLEM == Problem.maxcut:
        from rlsolver.methods.PIGNN.inference.inference_maxcut import run
        print("Loading MaxCut inference module...")
    elif PROBLEM == Problem.MIS:
        from rlsolver.methods.PIGNN.inference.inference_MIS import run
        print("Loading MIS inference module...")
    elif PROBLEM == Problem.graph_coloring:
        from rlsolver.methods.PIGNN.inference.inference_graph_coloring import run
        print("Loading Graph Coloring inference module...")
    else:
        raise ValueError(f"Unsupported problem type: {PROBLEM}")

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run the inference
    run()


def main():
    """Main entry point following ECO S2V pattern."""
    print_configuration()

    try:
        if TRAIN_INFERENCE == 0:  # Train mode
            print("Starting training mode...")
            dispatch_train_mode()
        elif TRAIN_INFERENCE == 1:  # Inference mode
            print("Starting inference mode...")
            dispatch_inference_mode()
        else:
            raise ValueError(f"Invalid TRAIN_INFERENCE value: {TRAIN_INFERENCE}. Must be 0 (train) or 1 (inference).")

    except ImportError as e:
        print(f"Error: Cannot import required module: {e}")
        print("Please ensure the corresponding train/inference scripts exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()