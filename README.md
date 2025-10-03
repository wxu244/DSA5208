# 🚀 Overview
- **nn.py**: Defines a simple feedforward neural network with configurable hidden dimension and activation function (`leaky_relu`, `relu`, `tanh`, `sigmoid`). Includes forward and backward propagation with gradient updates.
- **sgd.py**: Implements distributed training using MPI on the NYC Taxi dataset. Handles data loading, preprocessing, feature scaling, mini-batch training, and logging of results.
- **run_experiments.ps1**: PowerShell script for batch-running multiple experiments across different hyperparameters (activation, hidden units, batch size, etc.) using MPI.

# ⚡ Quick Start
First, update the CSV file path in `sgd.py` to match your local dataset location.
### 🔹 Run a single experiment
```console
mpiexec -n 4 python sgd.py --activation tanh --hidden 64 --batch 128 --epochs 100 --lr 0.001
```

### 🔹 Run batch experiments
Define multiple parameter settings in `run_experiments.ps1` (e.g., different activations, hidden dimensions, or batch sizes).  
```console
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
```
When all experiments are completed, the results will be appended to training_runs.jsonl, including parameter configurations, training/testing losses across epochs, and the final test RMSE for each run.