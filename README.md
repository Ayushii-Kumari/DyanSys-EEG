# DynaSys-EEG: Dynamical System-based EEG Analysis for Dementia Classification

A **dynamical systems + machine learning pipeline** for EEG-based dementia classification using **nonlinear state-space modeling, Takens embedding, and descriptor-based learning**.

It supports both:
- 🧪 Synthetic EEG data (for testing)
- 🧠 Real EEG datasets (Resting-state + Olfactory EEG)

---

## 🚀 Overview

This project models EEG signals as a **nonlinear dynamical system** instead of traditional feature engineering.
It extracts **system-level descriptors** from EEG and uses them for classification.

### 📌 Core Idea:
EEG → State Space Reconstruction → Dynamical System → Descriptor Vector → Classification

---

## 🧩 Pipeline Stages
```text
Phase 1 → Data Acquisition
Phase 2 → Signal Segmentation (5s windows, 50% overlap)
Phase 3 → Preprocessing (Bandpass + Normalization)
Phase 4 → State Space Reconstruction (Takens Embedding)
Phase 5 → Neural Dynamical System Learning (optional)
Phase 6 → Descriptor Computation (λ, H, D, E, T)
Phase 7 → Feature Vector Formation
Phase 8 → Classification (Multiple models)
Phase 9 → LOSO Validation
Phase 10 → Baseline Comparison
Phase 11 → Ablation Study
Phase 12 → Results Visualization
```
---

## 🧠 System Descriptors

Each EEG segment is converted into a **5-dimensional dynamical vector**:

- λ → Stability / Lyapunov-like behavior  
- H → Entropy (signal complexity)  
- D → Diffusion characteristics  
- E → Energy of system  
- T → Temporal structure  

Final feature vector:
Z = [λ, H, D, E, T]

---

## 📂 Project Structure

```text
DynaSys-EEG/
│
├── dynasys_eeg/
│   ├── pipeline.py
│   ├── configs.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── classification/
│   ├── evaluation/
│   └── utils/
│
├── convert_dataset/
│   ├── convert_primary_dataset.py
│   └── convert_secondary_dataset.py
│
├── data/
│   ├── primary/        # Resting EEG (AD, FTD, HC)
│   └── secondary/      # Olfactory EEG (AD, aMCI, HC)
│
├── results/
│   ├── descriptor_distributions.png
│   ├── confusion_matrix_loso.png
│   ├── method_comparison.png
│   ├── cross_dataset_performance.png
│
├── main.py
├── requirements.txt
└── README.md
```
---
## ⚙️ Installation

## 1️. Clone Repository
```bash 
git clone https://github.com/vanishkathakkar/DynaSys-EEG.git
cd DynaSys-EEG 
```

## 2️. Create Environment
```bash 
python -m venv venv
venv\Scripts\activate   # Windows
```

## 3️. Install Dependencies
```bash 
pip install -r requirements.txt
```

## How to Run

## 🔹 Run Full Pipeline (Synthetic Data)
```bash 
python main.py --mode synthetic
```

## 🔹 Run with Real EEG Data
```bash
python main.py --mode real \
--primary_dir data/primary \
--secondary_dir data/secondary
```

## 🔹 Run LOSO Evaluation Only
```bash
python main.py --mode synthetic --eval loso_only
```

## 🔹 Train Neural Dynamics Model (Optional)
```bash
python main.py --mode synthetic --train_dynamics
```

## 📊 Output Results

After execution, results are saved in:
```bash
results/
```

**Generated Visualizations:**

1. Descriptor Distribution plots
2. Confusion Matrix (LOSO)
3. Method Comparison (Models vs Baselines)
4. Cross-Dataset Performance
5. Ablation Study Results

---

## 🧪 Dataset Info

## 🧠 Primary Dataset (Resting EEG)

Classes: AD, FTD, HC
Used for main classification task

## 🧠 Secondary Dataset (Olfactory EEG)

Classes: AD, aMCI, HC
Used for cross-domain evaluation

## 📌 Key Features

1. Dynamical system modeling of EEG
2. Nonlinear state-space reconstruction (Takens theorem)
3. Multiple descriptor extraction (λ, H, D, E, T)
4. Multiple classifiers + baselines
5. LOSO cross-validation
6. Cross-dataset generalization testing
7. Ablation study support
8. Automated visualization pipeline

## 📈 Example Results (Observed)

**1. Primary Dataset (Resting EEG)**
Accuracy: ~29.55%
F1-score: ~0.21
→ Hard classification due to high variability

**2. Secondary Dataset (Olfactory EEG)**
Accuracy: ~45.71%
F1-score: ~0.41
→ Better separability

**3. Cross-Dataset (Resting → Olfactory)**
Accuracy: ~56.19%
→ Shows partial generalization ability

---

## 🧠 Interpretation

- Primary EEG is more noisy and less structured → harder classification
- Secondary EEG contains more stable patterns → better performance
- Cross-dataset performance shows model is learning general dynamical structure

---

## 🔮 Future Improvements
- Transformer-based dynamical modeling
- Better domain adaptation techniques
- Graph-based EEG modeling
- Real clinical dataset validation
- Deployment as web dashboard

The system follows 12 structured phases:
