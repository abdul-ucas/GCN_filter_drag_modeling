# GCN-Based Filtered Drag Modeling 

This repository provides an implementation of a Graph Convolutional Network (GCN)-based filtered drag model for gas–solid fluidized bed simulations.

The model and implementation is part of the supplentary material to the publication:
Paper title: “Graph convolutional network-based filtered drag model for coarse-grid TFM simulations of gas-solid fluidized beds”

Authors: Abdul Mateen, Qinyu Zhang, Fangpei Jin, Xiaoxing Liu

# The complete training framework will be released upon acceptance of the associated manuscript. However, the pretrained model can be used as described below.
---

## Repository Structure

```
├── gcn_filtered_drag_pretrained_model.ipynb          # jupyter notebook: source code used to load pretrained model 
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

Two pretrained models are provided based on the three-marker drift flux (DF) formulation — the best performing configuration identified across all trained variants:

| Model File | Particle System | Markers |
|------------|----------------|---------|
| `final_3_marker_gcn_DF_model_updated.pt` | Group 2: 75, 275, 300 μm | φ̄ₛ, ũ_slip,y, ∂p̄g/∂y |
| `geldart_B_3_marker_gcn_DF_model_updated.pt` | Group 1: 275, 300 μm (Geldart B) | φ̄ₛ, ũ_slip,y, ∂p̄g/∂y |

> **Note:** Multiple model configurations were trained across different input marker combinations and target variable formulations (drift flux and filtered drag force). The models provided here represent the best performing configuration based on our findings. 

---

## How to Use

### 1. Reload Pre-Trained Model (No Retraining Required)

However, the pretrained model can be used as described below.

```

Load the saved model directly and run inference on your own filtered data using gcn_filtered_drag_pre_trained_model.ipynb, for example:

```python
import torch

checkpoint = torch.load('saved_models/final_3_marker_gcn_DF_model_updated.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

> **Important:** You will need your own filtered TFM simulation data to use the pretrained model code. Filtered data from this study is available upon request as data pickle files (see Data Availability section below).

### 2. Training on New Datasets

To train the GCN model on a new dataset, provide your filtered simulation data in the graph format and run the k-fold cross-validation training loop inside `gcn_filter_drag_model.ipynb` (available upon acceptance of the associated manuscript). The training loop handles:

- Graph construction from filtered CFD data
- k-cross-validation training framework
- Model checkpointing based on best validation MSE
- Training history saving
- Embedding visuals and PCA

## Data Availability

 The filtered simulation datasets (data pickle files) used for training and evaluation are large files and are available from the corresponding author upon reasonable request.

---

## Citation

```
@article{abdul-ucas,
  title={Graph convolutional network-based filtered drag model for coarse-grid TFM simulations of gas-solid fluidized beds},
  author={Abdul Mateen et al.},
  journal={},
  year={2026}
  Zenodo: 10.5281/zenodo.20842931
}
```

---

## License

This repository is licensed under the BSD 3-Clause License. See `LICENSE` for details.
