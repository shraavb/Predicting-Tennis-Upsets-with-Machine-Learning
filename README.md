# Tennis Upset Prediction Model 🎾

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shraavb/Predicting-Tennis-Upsets-with-Machine-Learning/HEAD)

## Problem Statement
In professional tennis, predicting upsets (matches where a lower-ranked player defeats a higher-ranked one) is a challenging problem. This project builds a machine learning model to forecast the likelihood of an upset based on historical match and player statistics.

## Dataset
- **Source**: [ATP/WTA Tennis Data](https://github.com/JeffSackmann/tennis_atp) from Jeff Sackmann's repository
- **Coverage**: ATP and WTA tours from 2000-2025 (~138,000 matches)
- **Features**: Player rankings, match stats, surface type, tournament level, head-to-head records, and recent form
- **Data preprocessing**: Steps included in `notebooks/` and `src/data_loader.py`

## Approach / Model
- **Exploratory Data Analysis (EDA)** in Jupyter Notebooks (`/notebooks`)
- **Feature engineering** for player statistics, rankings, and surface-adjusted performance
- **Model training** using algorithms: Logistic Regression, Random Forest, XGBoost, and Neural Networks
- **Evaluation metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC

## Results
- **Best model**: **XGBoost** with 67% recall for upsets and 0.78 ROC-AUC
- **Upset rate**: 33% of matches result in upsets
- **Key predictors**: Ranking difference, head-to-head record, recent form, tournament level

## How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/shraavb/Predicting-Tennis-Upsets-with-Machine-Learning.git
   cd Predicting-Tennis-Upsets-with-Machine-Learning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks**:
   ```bash
   jupyter notebook notebooks/tennis_upset_prediction.ipynb
   ```

4. **Or run scripts**:
   ```bash
   python src/model.py
   ```

## Project Structure
```
tennis-upset-prediction/
├── data/                 # Raw and processed data
├── notebooks/            # Jupyter notebooks for EDA and analysis
│   └── tennis_upset_prediction.ipynb
├── src/                  # Python scripts
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── features.py       # Feature engineering
│   ├── model.py          # Model training
│   └── evaluate.py       # Model evaluation
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── .gitignore            # Git exclusions
```

## Future Work
- Hyperparameter tuning with Optuna
- Incorporating live match data for real-time predictions
- Deploying as a web app for interactive use
- Advanced ensemble methods and feature selection

## Video Presentation
[Watch the project presentation](https://drive.google.com/file/d/1jVA_WVHbcvK_gmuwNMx9xFhV_LMwHJ34/view?usp=share_link)

## License
CIS 545 Final Project - University of Pennsylvania
