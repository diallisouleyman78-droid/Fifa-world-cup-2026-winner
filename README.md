# FIFA World Cup 2026 Winner Prediction

A data science project that uses historical World Cup data, FIFA rankings, and team performance metrics to predict the winner of the 2026 FIFA World Cup.

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── results_1872_2026.csv                          # Historical international match results
│   │   └── wc_2026_48_teams_fifa_rank_change_corrected.csv  # 2026 WC qualified teams & FIFA rankings
│   └── interim/                                           # Intermediate transformed data
├── models/                                                # Saved trained models
├── notebooks/                                             # Jupyter notebooks for EDA & analysis
├── src/                                                   # Source scripts & utilities
├── main.py                                                # Main pipeline entry point
├── simulation.py                                          # Monte Carlo tournament simulation
├── .gitignore
├── requirements.txt                                       # Python dependencies
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

- **`results_1872_2026.csv`** — Historical international football match results from 1872 to 2026 (date, home/away teams, scores, tournament, city, country)
- **`wc_2026_48_teams_fifa_rank_change_corrected.csv`** — 48 qualified teams for the 2026 World Cup with FIFA ranking data
- FIFA Men's World Rankings
- Continental confederation affiliations
