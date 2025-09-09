# Predicting Tennis Upsets with Machine Learning

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shraavb/Predicting-Tennis-Upsets-with-Machine-Learning/HEAD)

Predicting when lower-ranked tennis players defeat higher-ranked opponents using machine learning on ATP and WTA match data (2000-2025).

## Overview

This project analyzes tennis match data to predict upsets - when underdogs beat favorites. We use features like ranking differences, head-to-head records, recent form, and tournament context to build predictive models.

## Key Results

- **Upset Rate**: 33% of matches result in upsets
- **Best Model**: XGBoost achieves 67% recall for upsets
- **Top Features**: Ranking difference, head-to-head record, recent form

## Models Tested

- Logistic Regression
- Random Forest  
- XGBoost
- Neural Network

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   ```bash
   jupyter notebook Final_Writeup.ipynb
   ```

## Files

- `Final_Writeup.ipynb`: Complete analysis
- `requirements.txt`: Dependencies
- `.gitignore`: Git exclusions

## Video Presentation

[Watch the project presentation](https://drive.google.com/file/d/1jVA_WVHbcvK_gmuwNMx9xFhV_LMwHJ34/view?usp=share_link)

CIS 545 Final Project - University of Pennsylvania
