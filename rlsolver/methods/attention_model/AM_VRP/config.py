class Config:
    # Model parameters
    EMBEDDING_DIM = 128
    N_ENCODE_LAYERS = 2
    N_HEADS = 8
    TANH_CLIPPING = 10.0
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 128  # per GPU batch size
    LR = 1e-4
    GRAD_CLIP = 1.0
    SAMPLES = 12800  # total samples across all GPUs
    
    # Problem parameters
    GRAPH_SIZE = 20
    
    # Baseline parameters
    WARMUP_EPOCHS = 1
    BASELINE_SAMPLES = 10000
    WARMUP_BETA = 0.8
    
    # Checkpoint parameters
    CHECKPOINT = None
    BASELINE_CHECKPOINT = None
    
    # Distributed parameters
    WORLD_SIZE = -1  # will be set by environment
    RANK = -1  # will be set by environment
    LOCAL_RANK = -1  # will be set by environment
    BACKEND = 'nccl'
    
    # Other parameters
    DEVICE = 'cuda'
    SEED = 1234
    VAL_SIZE = 10000
    FILENAME = None
    
    # Display parameters
    BATCH_VERBOSE = 1000
    VAL_BATCH_SIZE = 1000