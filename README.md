Execution Environment and Instructions
All experiments in this project were conducted using Python and PyTorch. The core architecture implements the Residual Flow model as proposed by Chen et al. (2021), enabling exact likelihood estimation via a chain of invertible residual blocks. All experiments were executed on CUDA-compatible GPUs for efficient training.

To ensure reproducibility, we provide a Conda environment specification (residualflow_env.yml) that lists all required dependencies. The environment can be created and activated using the following commands:

conda env create -f residualflow_env.yml
conda activate residualflow

This environment includes Python (≥3.8), PyTorch (≥1.10), torchvision, matplotlib, numpy, tqdm, imageio, and PyYAML. These packages are essential for training, data preprocessing, visualization, and configuration management.

The codebase is modular and organized into the following key components:

main.py: Entry point for training and evaluation. This script loads the configuration, initializes the model, and launches the training process.

train.py: Implements the training loop, loss computation, and model evaluation routines.

residualflow.py: Constructs the overall model by composing a sequence of invertible transformations.

iresblock.py: Defines the invertible residual blocks used in the model. These blocks employ spectral normalization to enforce Lipschitz continuity.

container.py: Provides a wrapper for chaining multiple invertible modules with consistent forward and inverse interfaces.

normalization.py: Contains normalization layers such as Moving BatchNorm, adapted to maintain invertibility.

elemwise.py: Implements channel-wise affine transformations such as zero-mean shifting and scaling.

visualize_flow.py: Offers tools for visualizing the learned flow transformations and grid deformations.

To begin training the model (e.g., on CIFAR-10), simply run:

python main.py

The training process automatically logs relevant metrics and saves checkpoints to the default output directory. Hyperparameters such as learning rate, batch size, number of residual blocks, and optimizer settings can be configured directly in main.py or by modifying the codebase as needed. The model is saved using PyTorch’s serialization functions and can be restored for evaluation or fine-tuning.

This configuration ensures the reproducibility of results with minimal effort, while allowing for flexible extension of the architecture and training procedure.
