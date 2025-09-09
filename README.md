# Tennis Upset Prediction Model 🎾

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shraavb/Predicting-Tennis-Upsets-with-Machine-Learning/HEAD)

Predicting tennis upsets using machine learning on ATP/WTA match data (2000-2025).

## Problem
Build a model to predict when lower-ranked players defeat higher-ranked opponents in professional tennis matches.

## Dataset
- **Source**: [ATP/WTA Tennis Data](https://github.com/JeffSackmann/tennis_atp)
- **Size**: ~138,000 matches from 2000-2025
- **Features**: Rankings, match stats, surface type, tournament level, head-to-head records

## Results
- **Best Model**: XGBoost with 67% recall for upsets
- **Upset Rate**: 33% of matches
- **Key Features**: Ranking difference, head-to-head record, recent form

## Quick Start

1. **Clone & Install**:
   ```bash
   git clone https://github.com/shraavb/Predicting-Tennis-Upsets-with-Machine-Learning.git
   cd Predicting-Tennis-Upsets-with-Machine-Learning
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   jupyter notebook notebooks/tennis_upset_prediction.ipynb
   # OR
   python src/model.py
   ```

## Project Structure
```
├── data/                 # Raw and processed data
├── notebooks/            # Jupyter notebooks
├── src/                  # Python scripts
│   ├── data_loader.py    # Data loading
│   ├── features.py       # Feature engineering
│   ├── model.py          # Model training
│   └── evaluate.py       # Model evaluation
└── requirements.txt      # Dependencies
```

## Video
[Watch the presentation](https://drive.google.com/file/d/1jVA_WVHbcvK_gmuwNMx9xFhV_LMwHJ34/view?usp=share_link)

CIS 545 Final Project - University of Pennsylvania
