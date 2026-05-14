# GCN-Based Filtered Drag Modeling 

This repository provides an implementation of a Graph Convolutional Network (GCN)-based filtered drag model for gas–solid fluidized bed simulations.

The model is trained using graph-structured filtered Two-Fluid Model (TFM) simulation data to predict drift flux, which is then used to calculate filter drag force.

---

## Repository Structure

```
├── gcn_filter_drag_model.ipynb          # Main notebook: data preparation, graph construction, GCN training and evaluation
├── saved_models/
│   ├── final_3_marker_gcn_DF_model_updated.pt           # Best model: Group 2 (75, 275, 300 μm)
│   └── geldart_B_3_marker_gcn_DF_model_updated.pt       # Group 1: Geldart B particles (275, 300 μm)
├── saved_training_history/              # Training and validation loss history for all folds
└── LICENSE
```

---

## Requirements

```
Python        (we used Python 3.11.13 for development and testing)
PyTorch       (deep learning framework)
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook
```

Install dependencies:

```bash
!pip install torch torch-geometric numpy pandas matplotlib scikit-learn scipy
```

---

## Saved Models

Two pre-trained models are provided based on the three-marker drift flux (DF) formulation — the best performing configuration identified across all trained variants:

| Model File | Particle System | Markers |
|------------|----------------|---------|
| `final_3_marker_gcn_DF_model_updated.pt` | Group 2: 75, 275, 300 μm | φ̄ₛ, ũ_slip,y, ∂p̄g/∂y |
| `geldart_B_3_marker_gcn_DF_model_updated.pt` | Group 1: 275, 300 μm (Geldart B) | φ̄ₛ, ũ_slip,y, ∂p̄g/∂y |

> **Note:** Multiple model configurations were trained across different input marker combinations and target variable formulations (drift flux and filtered drag force). The models provided here represent the best performing configuration based on our findings. 

---

## How to Use

### 1. Reload Pre-Trained Model (No Retraining Required)

Open `gcn_filter_drag_model.ipynb` and navigate to the section:

```
# Reload the saved model without retraining the model
```

Load the saved model directly and run inference on your own filtered data:

```python
import torch

checkpoint = torch.load('saved_models/final_3_marker_gcn_DF_model_updated.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

> **Important:** You will need your own filtered TFM simulation data to use the pre-trained model. Filtered data from this study is available upon request as data pickle files (see Data Availability section below).

### 2. Retrain on New Datasets

To train the GCN model on a new dataset, provide your filtered simulation data in the required graph format and run the k-fold cross-validation training loop inside `gcn_filter_drag_model.ipynb`. The training loop handles:

- Graph construction from filtered CFD data
- 8-fold cross-validation with early stopping
- Model checkpointing based on best validation MSE
- Training history saving

## Data Availability

The source code, saved model weights, and training history are publicly available in this repository. The filtered simulation datasets (data pickle files) used for training and evaluation are large files and are available from the corresponding author upon reasonable request.

---

## Citation

```
@article{mateen2025gcn,
  title={Development of graph convolutional network-based filtered drag closure for coarse grid two-fluid model simulation of gas-solid flows},
  author={Abdul Mateen et al.},
  journal={},
  year={2026}
}
```

---

## License

This repository is licensed under the BSD 3-Clause License. See `LICENSE` for details.
