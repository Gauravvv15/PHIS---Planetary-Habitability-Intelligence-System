# PHIS — Planetary Habitability Intelligence System
### Machine Learning Framework for Exoplanet Habitability Classification and Earth Similarity Prediction

## Overview
[cite_start]PHIS (Planetary Habitability Intelligence System) is an astronomy-focused machine learning project designed to evaluate exoplanets for potential habitability and predict their Earth Similarity Index (ESI) using astrophysical, planetary, orbital, and stellar parameters[cite: 307].

[cite_start]By integrating NASA Exoplanet Archive data with Planetary Habitability Laboratory (PHL) datasets, PHIS creates a robust predictive framework capable of[cite: 308]:

- [cite_start]Classifying whether an exoplanet is potentially habitable [cite: 309]
- [cite_start]Predicting Earth Similarity Index (ESI) [cite: 309]
- [cite_start]Engineering scientific features using astrophysical formulas [cite: 309]
- [cite_start]Simulating real-world planetary habitability analysis [cite: 309]

[cite_start]This project combines data science, astronomy, and machine learning into a portfolio-grade scientific system[cite: 309].

---

## Key Features

### Data Integration
- [cite_start]NASA Exoplanet Archive dataset [cite: 310]
- [cite_start]Planetary Habitability Laboratory (PHL) dataset [cite: 310]
- [cite_start]Cross-dataset feature merging using cleaned planetary identifiers [cite: 310]

### Advanced Feature Engineering
[cite_start]PHIS calculates critical astrophysical features including[cite: 310]:

- [cite_start]Orbital Period (Kepler’s Third Law) [cite: 310]
- [cite_start]Stellar Flux [cite: 310]
- [cite_start]Equilibrium Temperature [cite: 310]
- [cite_start]Periastron & Apastron distances [cite: 310]
- [cite_start]Effective orbital distance [cite: 310]
- [cite_start]Stellar surface gravity (log g) [cite: 310]
- [cite_start]Conservative and optimistic habitable zone boundaries [cite: 310]
- [cite_start]Abiogenesis zone estimation [cite: 310]
- [cite_start]Tidal locking probability [cite: 311]
- [cite_start]Planet type classification (Terran, SuperTerran, Neptunian, Jovian) [cite: 311]

### Machine Learning Models
#### Habitability Classification:
- [cite_start]XGBoost Classifier [cite: 312]
- [cite_start]SMOTE for class imbalance correction [cite: 312]
- [cite_start]High recall optimization for habitable planets [cite: 312]

#### ESI Prediction:
- [cite_start]XGBoost Regressor [cite: 312]
- [cite_start]Regression analysis for Earth similarity scoring [cite: 312]

### Performance
#### Classification:
- [cite_start]Accuracy: ~99.6% [cite: 312]
- [cite_start]High recall for habitable class [cite: 312]
- [cite_start]Strong macro F1 score [cite: 312]

#### Regression:
- [cite_start]R² Score: ~0.94 [cite: 312]
- [cite_start]Low MAE [cite: 312]
- [cite_start]Reliable ESI prediction consistency [cite: 312]

---

## Project Structure
```bash
PHIS/
│── data/                     # Raw NASA + PHL datasets
│── models/                   # Saved ML models, encoders, scaler
│   ├── xgb_habitability_model.pkl
│   ├── xgb_esi_model.pkl
│   ├── scaler.pkl
│   ├── le_disc.pkl
│   ├── le_ptype.pkl
│   └── df_cols.pkl
│
│── source/
│   ├── preprocessing_df.py   # Training data preprocessing
│   ├── preprocessing_in.py   # User input preprocessing + feature engineering
│   ├── train.py              # Model training pipeline
│   ├── predict.py            # Prediction + scientific interpretation
│   └── user.py               # User interaction system
│
│── visuals/                  # EDA, feature importance, confusion matrix, residuals
│── main.py                   # Main execution file
│── requirements.txt
│── LICENSE
│── README.md
└── .gitignore
```
---

## Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/PHIS.git
cd PHIS
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Requirements
```txt
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
requests
```

---

## How It Works

### Training Pipeline
1. Load NASA + PHL datasets
2. Clean and merge datasets
3. Feature engineering
4. Missing value imputation
5. Label encoding
6. Robust scaling
7. Train/test split
8. SMOTE balancing
9. Train XGBoost classifier and regressor
10. Save trained models

### Prediction Pipeline
User inputs:
- Planet radius
- Planet mass
- Orbital distance
- Eccentricity
- Stellar temperature
- Stellar mass
- Stellar luminosity
- Stellar metallicity
- Stellar age
- Stellar radius

PHIS then:
- Engineers scientific features
- Applies scaler/encoders
- Predicts habitability probability
- Predicts Earth Similarity Index
- Provides scientific interpretation

---

## Example Output
```bash
Planet Classification: HABITABLE
Confidence Level: 94.92%

Earth Similarity Index (ESI): 0.733
Habitability Category: Moderately Earth-Like
```

---

## Scientific Significance
PHIS is not just a machine learning project — it represents a practical simulation of computational astrobiology.

### Applications:
- Exoplanet candidate prioritization
- Habitability analysis
- Astrobiological research support
- Astronomy-focused ML portfolio development
- Future integration with live NASA databases

---

## Visualizations Included
- Confusion Matrix
- Feature Importance (Habitability)
- Feature Importance (ESI)
- Residual Error Analysis
- EDA plots

---

## Future Improvements
### Planned Upgrades:
- Streamlit web deployment
- Live NASA API integration
- Interactive dashboard
- PDF scientific report generation
- Planet comparison tools
- Deep learning enhancements
- Automated candidate ranking system

---

## Project Strengths
- Real scientific domain application
- High-performance ML architecture
- Strong feature engineering
- Production-ready model storage
- User interaction support
- Research scalability

---

## Author
### Gaurav
AIML Student | Data Science & Astronomy Enthusiast

Focused on building machine learning systems for:
- Astronomy
- Planetary science
- Scientific AI applications

---

## License
This project is licensed under the MIT License.

---

## Final Note
PHIS demonstrates how machine learning can be combined with astrophysical science to build intelligent systems capable of exploring one of humanity’s greatest questions:

### *Could another Earth exist?*

