# Cancer Tumour Model in Presence of Magnetic Nanoparticles

**Authors:** Yeganeh Abdollahinejad, Amit K Chattopadhyay, Gillian Pearce, Angelo Sajan Kalathil

---

## Overview

This repository contains all data, code, and scripts for the paper:

> **Cancer Tumour Model in Presence of Magnetic Nanoparticles**

We present a comprehensive computational framework that integrates:
- **Mechanistic mathematical modeling** (PDEs/ODEs) of nanoparticle-tumor interactions
- **Bayesian parameter estimation** using Markov Chain Monte Carlo (MCMC)
- **Machine learning models** (Gaussian Process Regression, Random Forest, Gradient Boosting Machine) for predictive tumor growth analysis

All code and data are provided for full reproducibility of the results and figures in the manuscript.

---

## Directory Structure

```
cancer-modeling--main/
│
├── datasets/                # All experimental and processed data files
│   ├── Qi70/
│   ├── Qi100/
│   ├── saline85/
│   ├── untreated85/
│   └── ... (other groups)
│
├── src/
│   ├── estimation/          # Parameter estimation, mechanistic modeling, MCMC
│   │   ├── bayesian-inference.py
│   │   ├── mcmc.py
│   │   ├── integration.py
│   │   └── ...
│   ├── visualization/       # All plotting and figure generation scripts
│   │   ├── math_bayes_figures.py
│   │   ├── paper_figures.py
│   │   └── ...
│   └── utils/               # Helper scripts/utilities
│
├── figures/                 # All generated figures for the paper
│   ├── math_bayes_multi_panel.png
│   ├── qi70_model_fit.png
│   ├── qi100_model_fit.png
│   ├── rf_all_treatments.png
│   ├── gpr_all_treatments.png
│   ├── gbm_all_treatments.png
│   └── ...
│
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── paper_draft.tex          # LaTeX manuscript (optional)
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd cancer-modeling--main
```

### 2. Set Up the Python Environment

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Data Files

All required data files are in the `datasets/` directory, organized by experiment/treatment group.

---

## Reproducing the Results

### **A. Mechanistic Model & Bayesian Inference**

- Scripts:  
  - `src/estimation/bayesian-inference.py`  
  - `src/estimation/mcmc.py`  
  - `src/estimation/integration.py`

- **What they do:**  
  - Estimate model parameters using 75% of the data (training set)
  - Simulate the model using estimated parameters
  - Predict tumor growth for the remaining 25% (test set)
  - Generate model fit and parameter comparison plots

- **How to run:**
  ```bash
  python src/estimation/bayesian-inference.py
  # or
  python src/estimation/mcmc.py
  ```

### **B. Visualization and Figure Generation**

- Scripts:  
  - `src/visualization/math_bayes_figures.py`  
  - `src/visualization/paper_figures.py`

- **What they do:**  
  - Generate all figures for the paper, including:
    - Data split visualizations (100%, 75% train, 25% test)
    - Model fit overlays (model prediction vs. test data)
    - Bar plots of estimated parameters
    - Machine learning model comparisons (RF, GPR, GBM, etc.)

- **How to run:**
  ```bash
  python src/visualization/math_bayes_figures.py
  python src/visualization/paper_figures.py
  ```

- **All figures will be saved in the `figures/` directory.**

---

## Key Features

- **Mechanistic Model:**  
  - PDE/ODE system for nanoparticle diffusion, binding, internalization, and tumor cell dynamics
- **Bayesian Inference:**  
  - Parameter estimation using MCMC (emcee)
  - Uncertainty quantification and convergence diagnostics
- **Machine Learning:**  
  - Gaussian Process Regression (GPR)
  - Random Forest (RF)
  - Gradient Boosting Machine (GBM)
  - All models trained/tested on a 75/25 split, with performance metrics and comparison plots

---

## Requirements

All dependencies are listed in `requirements.txt`.  
Key packages include:
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- emcee
- corner
- tensorflow
- openpyxl

Install with:
```bash
pip install -r requirements.txt
```

---

## Data and Code Availability

All data and modeling scripts are available in this repository. For any questions, please contact the authors.

---

## Citation
If you use this code or data, please cite our paper:

> Yeganeh Abdollahinejad, Amit K Chattopadhyay, Gillian Pearce, Angelo Sajan Kalathil. "Cancer Tumour Model in Presence of Magnetic Nanoparticles." (Year, Journal, etc.)

---
