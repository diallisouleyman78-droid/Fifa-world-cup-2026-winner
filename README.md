# FIFA World Cup 2026 Winner Prediction

A data science project that uses historical World Cup data, FIFA rankings, and team performance metrics to predict the winner of the 2026 FIFA World Cup.

## Project Structure

```
├── data/                   # Collected datasets
│   ├── raw/                # Raw scraped data
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # Jupyter notebooks
│   └── analysis.ipynb      # Main analysis notebook
├── src/                    # Source scripts
│   └── data_collection.py  # Data scraping utilities
├── requirements.txt        # Python dependencies
└── README.md
```

## Methodology

1. **Data Collection** — Historical World Cup match results (1930–2022), FIFA rankings, and continental confederation data
2. **Exploratory Data Analysis** — Trends in World Cup winners, home advantage, confederation strength
3. **Feature Engineering** — Win rates, ranking differentials, historical performance, goal scoring records
4. **Modeling** — Multiple ML models (Logistic Regression, Random Forest, XGBoost) to predict match outcomes
5. **Simulation** — Monte Carlo simulation of the 2026 tournament bracket

## Setup

```bash
conda activate "/home/diallo/Documents/Data science projects/Fifa world cup 2026 winner/venv"
pip install -r requirements.txt
```

## Data Sources

- Historical World Cup match results (1930–2022)
- FIFA Men's World Rankings
- Continental confederation affiliations
- Team performance metrics (goals scored/conceded, win rates)
