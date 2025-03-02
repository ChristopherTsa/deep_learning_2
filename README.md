# Deep Learning 2 - RBMs, DBNs, and DNNs

This repository contains implementations of Restricted Boltzmann Machines (RBMs), Deep Belief Networks (DBNs), and Deep Neural Networks (DNNs), along with experiments on MNIST and Binary AlphaDigits datasets.

## Project Structure

```
deep_learning_2/
├── models/                 # Model implementations
│   ├── __init__.py         # Model exports
│   ├── rbm.py              # RBM, PersistentRBM, and TRBM implementations
│   ├── dbn.py              # DBN implementation
│   └── dnn.py              # DNN implementation
├── utils/                  # Utility functions
│   ├── __init__.py         # Utility exports
│   ├── data_utils.py       # Data loading and preprocessing
│   └── visualization.py    # Plotting and visualization functions
├── data/                   # Data directory (created automatically)
│   ├── mnist.pkl           # Cached MNIST dataset (created after first run)
│   └── binaryalphadigs.mat # Binary AlphaDigits dataset (to be downloaded)
├── results/                # Results directory (created automatically)
│   ├── models/             # Saved models
│   └── plots/              # Generated plots and visualizations
├── main_rbm_alpha.py       # RBM experiments with Binary AlphaDigits
├── main_dbn_alpha.py       # DBN experiments with Binary AlphaDigits
├── main_dnn_mnist.py       # DNN experiments with MNIST
└── README.md               # This file
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- Joblib

## Data Download Instructions

### MNIST Dataset
The MNIST dataset will be automatically downloaded from OpenML when running the scripts. It will be cached in `data/mnist.pkl` for faster subsequent loading.

### Binary AlphaDigits Dataset
This dataset must be downloaded manually from Kaggle:

1. Go to [Binary Alpha-Digits on Kaggle](https://www.kaggle.com/datasets/angevalli/binary-alpha-digits)
2. Download the `binaryalphadigs.mat` file
3. Place the file in the `data/` directory of this project:
   ```
   mkdir -p data
   mv ~/Downloads/binaryalphadigs.mat data/
   ```

## Usage

### Training RBMs on Binary AlphaDigits

```bash
python main_rbm_alpha.py
```

This script:
- Loads the Binary AlphaDigits dataset (digits 0-9)
- Trains an RBM on the data
- Generates samples and reconstructions
- Saves the model and visualizations in the `results/` directory

### Training DBNs on Binary AlphaDigits

```bash
python main_dbn_alpha.py
```

This script:
- Loads the Binary AlphaDigits dataset (digits 0-9)
- Trains a 2-layer DBN on the data
- Generates samples and reconstructions
- Saves the model and visualizations in the `results/` directory

### Training DNNs on MNIST

```bash
python main_dnn_mnist.py
```

This script:
- Loads the MNIST dataset
- Performs experiments comparing pretrained vs. randomly initialized DNNs:
  - Effect of network depth (number of layers)
  - Effect of network width (neurons per layer)
  - Effect of training set size
- Trains an optimal model with the best configuration
- Visualizes results and saves models in the `results/` directory

## Model Classes

### RBM Models (`models/rbm.py`)
- `RBM`: Basic Restricted Boltzmann Machine
- `PersistentRBM`: RBM with Persistent Contrastive Divergence
- `TRBM`: RBM with temperature parameter

### DBN Model (`models/dbn.py`)
- `DBN`: Deep Belief Network, composed of stacked RBMs

### DNN Model (`models/dnn.py`)
- `DNN`: Deep Neural Network, supports initialization from a trained DBN

## Utils and Visualization

### Data Utils (`utils/data_utils.py`)
- `load_mnist`: Load and preprocess MNIST dataset
- `load_binary_alphadigits`: Load Binary AlphaDigits dataset

### Visualization (`utils/visualization.py`)
- `plot_losses`: Plot training/validation loss curves
- `plot_comparison`: Compare performance metrics
- `display_binary_images`: Display binary image grids
- `display_weights`: Visualize RBM weights

## Expected Outputs

After running the scripts, you'll find:

1. Trained models in `results/models/`
2. Visualizations in `results/plots/`, including:
   - Weight visualizations
   - Generated samples
   - Reconstructions
   - Error plots
   - Performance comparisons