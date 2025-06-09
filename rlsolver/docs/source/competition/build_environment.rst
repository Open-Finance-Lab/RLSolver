Build Environment
=================

This section explains how to set up the software and hardware environment required to run RLSolver Competition 2025 code.

1. **Operating System**  
   - Ubuntu 20.04 (recommended) or Windows 10 (WSL is supported).  
   - If you use Windows, install WSL2 with Ubuntu for GPU support.

2. **Hardware Requirements**  
   - NVIDIA GPU with CUDA capability (compute capability ≥ 6.0).  
   - At least 8 GB of RAM.  
   - At least 20 GB of free disk space for data and models.

3. **Software Dependencies**  

4. **Creating a Python Virtual Environment**  
- Using Conda:
  ```
  conda create -n rlsolver python=3.8
  conda activate rlsolver
  ```
- Or using `venv`:
  ```
  python3 -m venv rlsolver_env
  source rlsolver_env/bin/activate
  ```

5. **Installing Python Packages**  
- After activating the virtual environment:
  ```
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
  pip install scikit-learn networkx pandas matplotlib numpy
  pip install tqdm
  ```
- Verify CUDA is available:
  ```python
  import torch
  print(torch.cuda.is_available())  # should print True if GPU is detected
  ```

6. **(Optional) Docker Container**  
- A `Dockerfile` is provided to encapsulate all dependencies.
- Build the Docker image:
  ```
  docker build -t rlsolver:latest .
  ```
- Run a container with GPU enabled:
  ```
  docker run --gpus all -v $(pwd):/workspace -it rlsolver:latest /bin/bash
  ```
- Inside the container, you can run training/testing commands as described in other sections.

7. **Directory Structure After Setup**  