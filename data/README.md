# Data Directory

This directory contains the tennis match data used for the upset prediction model.

## Data Source

The data is sourced from Jeff Sackmann's tennis data repository:
- **ATP Data**: https://github.com/JeffSackmann/tennis_atp
- **WTA Data**: https://github.com/JeffSackmann/tennis_wta

## Data Description

- **Coverage**: ATP and WTA tours from 2000-2025
- **Size**: ~138,000 matches
- **Features**: Player rankings, match statistics, tournament details, surface types, head-to-head records

## Data Processing

The raw data is automatically downloaded and processed by the scripts in `src/data_loader.py`. The processed data includes:

- Restructured matches (favorite vs underdog format)
- Engineered features (ranking differences, recent form, head-to-head records)
- Encoded categorical variables
- Temporal leakage prevention

## Usage

To download and process the data, run:

```python
from src.data_loader import load_tennis_data, restructure_data
from src.features import prepare_features

# Load raw data
data = load_tennis_data(range(2000, 2025))

# Restructure to prevent temporal leakage
restructured = restructure_data(data)

# Engineer features
features = prepare_features(restructured)

# Save processed data
features.to_csv('data/tennis_data_processed.csv', index=False)
```

## File Structure

- `tennis_data_processed.csv`: Final processed dataset ready for modeling
- `README.md`: This file
