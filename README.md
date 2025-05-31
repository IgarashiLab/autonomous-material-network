# Autonomous Material Network

A Bayesian Optimization system for autonomous materials discovery and optimization, developed for research by Yoshida, Iwabuchi, Igarashi, and Iwasaki.

## Overview

This repository implements a multi-objective Bayesian Optimization network for materials research, featuring:

- **Multi-target optimization**: Simultaneous optimization of three material properties (CT, MM, SP)
- **Transfer learning**: Knowledge transfer between different optimization systems
- **Ensemble prediction**: Robust predictions using ensemble methods
- **Parallel execution**: Concurrent optimization across multiple systems
- **Docker containerization**: Reproducible environment with CUDA support

## System Architecture

The system consists of three Bayesian Optimization modules:

- **BO_system_CT**: Optimization for CT (Computed Tomography) properties
- **BO_system_MM**: Optimization for MM (Mechanical Modulus) properties  
- **BO_system_SP**: Optimization for SP (Specific Properties)

Each system supports:
- Transfer learning from other systems
- Ensemble prediction with uncertainty quantification
- Independent dataset initialization and management

## Requirements

- Docker with NVIDIA Container Toolkit (for GPU support)
- CUDA 11.0.3 compatible GPU (optional but recommended)
- Make utility

## Quick Start

### 1. Environment Setup (Docker Container Build)

```bash
make
```

This command builds the Docker image with all required dependencies including:
- Python 3 with scientific computing libraries
- PyTorch 2.0.1 with CUDA support
- Scikit-learn, Pandas, Matplotlib
- GPy for Gaussian Process modeling
- MLflow for experiment tracking

### 2. Container Access

```bash
make attach
```

Attach to the running Docker container to access the workspace.

### 3. Running Optimization

Navigate to the workspace and run the network optimization:

```bash
cd /workspace/BO_system_network2
python3 run_network2.py -p network9_6 -n 10
```

Parameters:
- `-p, --network_path`: Path to the network configuration
- `-n, --run_num`: Number of optimization runs (default: 10)

## Project Structure

```
autonomous-material-network/
├── Dockerfiles/
│   └── Dockerfile_cuda_v2          # CUDA-enabled container definition
├── workspace/
│   └── BO_system_network2/
│       ├── run_network2.py         # Main execution script
│       ├── network9_6/             # Network configuration v9.6
│       │   ├── BO_system_CT/       # CT optimization system
│       │   ├── BO_system_MM/       # MM optimization system
│       │   ├── BO_system_SP/       # SP optimization system
│       │   ├── initialize_BO_system.py
│       │   └── run_networkBO.py
│       ├── network10_6/            # Network configuration v10.6
│       └── *.ipynb                 # Analysis notebooks
├── Makefile                        # Docker management commands
└── README.md                       # This file
```

## Usage

### System Initialization

Initialize a specific optimization system:

```bash
python3 initialize_BO_system.py -s BO_system_CT -o CT -n 10 -r 42
```

Parameters:
- `-s, --system_path`: Target system path
- `-o, --obj_name`: Optimization objective (CT/MM/SP)
- `-n, --observed_num`: Number of initial observed data points
- `-r, --random_seed`: Random seed for reproducibility

### Network Optimization

Run the complete network optimization:

```bash
python3 run_networkBO.py
```

This script reads configuration from `setting.yaml` and coordinates optimization across all three systems with transfer learning.

### Parallel Execution

Execute multiple optimization runs in parallel:

```bash
python3 run_network2.py -p network9_6 -n 20
```

Results are automatically collected and organized in the `result_network9_6/` directory.

## Configuration

The system uses YAML configuration files (`setting.yaml`) to control:

- Optimization parameters for each system
- Transfer learning intervals
- Processing time schedules
- Random seeds and reproducibility settings

## Output Files

The system generates several types of output:

- `observed_target.csv`: Observed optimization targets
- `unobserved_prediction.csv`: Predictions for unobserved materials
- `true_estimate_*.csv`: True value estimations at different stages
- Ensemble prediction results with uncertainty quantification

## Docker Commands

Available Make targets:

```bash
make create    # Build image and create container
make build     # Build Docker image only
make run       # Create and start container
make attach    # Attach to running container
make start     # Start stopped container
make stop      # Stop running container
make exec      # Execute bash in container
make allrm     # Remove container and image
```

## Dependencies

The Docker environment includes:

- **Base**: NVIDIA CUDA 11.0.3 on Ubuntu 20.04
- **Python Libraries**:
  - PyTorch 2.0.1 + TorchVision 0.15.2
  - Scikit-learn, Pandas, NumPy, Matplotlib
  - GPy (Gaussian Processes)
  - MLflow (Experiment tracking)
  - PyTorch Lightning

## Research Context

This system implements autonomous materials discovery methods for accelerating the development of new materials with desired properties. The multi-objective Bayesian optimization approach enables efficient exploration of the materials design space while leveraging transfer learning to share knowledge between related optimization tasks.

## License

See [LICENSE](LICENSE) file for details.
