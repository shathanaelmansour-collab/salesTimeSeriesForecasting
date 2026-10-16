# 📈 Retail Sales Forecasting with Prophet

**End-to-end time-series forecasting of company-wide daily sales using the Ecuador Favorita dataset, with an interactive Streamlit dashboard.**

### 🌐 [Launch the Live Forecasting Dashboard](https://salestimeseriesforecasting-egqvy7htlehhkvnzf9haxc.streamlit.app/)

## Project overview

This project investigates historical retail sales patterns and forecasts future **total daily sales across all stores and product families**. It covers data preparation, exploratory data analysis (EDA), baseline modeling, Prophet forecasting, holiday-effect experiments, time-series cross-validation, hyperparameter tuning, and deployment.

The deployed application loads a previously trained Prophet model and generates forecasts for a user-selected horizon. It does **not** retrain the model or accept store-level, product-level, or promotional inputs.

## Dataset

- **Source:** [Kaggle — Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Retailer:** Corporación Favorita, Ecuador
- **Training data:** Approximately 3 million sales records covering **January 1, 2013 – August 15, 2017**
- **Target:** Daily aggregated sales, `y`, indexed by date, `ds`
- **Granularity:** Company-wide daily sales, aggregated across stores and product families

The analysis also examines holiday information. The original competition files are not included in this repository; obtain them from Kaggle to reproduce the notebook.

## Project workflow

1. **Data preparation:** Inspect data quality, convert dates, aggregate sales by day, and prepare Prophet-compatible `ds` and `y` columns.
2. **Exploratory analysis:** Examine the overall trend, weekly seasonality, unusually low sales days, and possible holiday effects.
3. **Chronological validation:** Reserve the final **90 days** as a holdout test set to avoid random-split leakage.
4. **Baseline:** Evaluate a seasonal-naive forecast.
5. **Prophet experiments:** Compare a standard weekly/yearly seasonal Prophet model with a holiday-enhanced variant.
6. **Cross-validation and tuning:** Evaluate rolling historical forecasts and tune Prophet's changepoint and seasonality prior scales.
7. **Final training:** Refit the selected tuned Prophet configuration on all available historical daily sales.
8. **Deployment:** Serialize the final model to JSON and serve interactive predictions using Streamlit.

## Model performance

**Historical 90-day holdout test** (lower is better):

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal naive baseline | 93,467.95 | 130,808.95 | 10.93% |
| Prophet V1 | 71,931.30 | 94,340.65 | 8.41% |
| Prophet V2 + holidays | 69,367.87 | 90,866.63 | **8.11%** |
| Tuned Prophet V1 | 72,122.47 | 94,342.70 | 8.43% |

The holiday-enhanced model performed best on this **specific holdout period**. The deployed model instead uses the **tuned Prophet V1 configuration**, selected based on historical cross-validation and refitted on the full dataset. Therefore, **8.11% WAPE is not the reported test performance of the deployed configuration**.

Historical cross-validation WAPE results were approximately **12.57%** for Prophet V1, **12.61%** for Prophet V2 with holidays, and **12.47%** for tuned Prophet V1. The selected tuning parameters were `changepoint_prior_scale=0.05` and `seasonality_prior_scale=1.0`.

**Metric definitions:** MAE = Mean Absolute Error; RMSE = Root Mean Squared Error; WAPE = Weighted Absolute Percentage Error.

## Interactive dashboard

🔗 **[Open the Streamlit App](https://salestimeseriesforecasting-egqvy7htlehhkvnzf9haxc.streamlit.app/)**

Visitors can:

- Select a **7–180 day** forecast horizon.
- Adjust how much historical data appears in the charts.
- Generate daily predictions using the saved Prophet model.
- View forecast uncertainty intervals and summary statistics.
- Compare the historical holdout results of the evaluated approaches.
- Download forecasted daily sales as a CSV file.

**Forecast reference date:** The dataset ends on **August 15, 2017**, so all app forecasts begin on **August 16, 2017**. These are historical-data projections for demonstration, **not current sales forecasts**. Horizons longer than 90 days were not evaluated in the project's holdout test.

## Repository structure

```text
salesTimeSeriesForecasting/
├── App.py                         # Interactive Streamlit dashboard
├── requirements.txt               # Application dependencies
├── .gitignore
├── README.md
├── data/
│   └── daily_sales.csv            # Aggregated historical daily sales
├── models/
│   └── prophet_model.json         # Serialized fitted Prophet model
└── notebooks/
    └── TimeSeriesForecasting.ipynb # Exploration, evaluation and training
```

## Run locally

**Prerequisites:** Python and the dependencies in `requirements.txt`. Use a Python version supported by the pinned Prophet version (the cloud deployment was configured for Python 3.12).

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Then install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run App.py
```

Ensure `data/daily_sales.csv` and `models/prophet_model.json` exist in the paths shown above. The app uses the serialized model and does not need to rerun the training notebook. On computers with enforced application-control restrictions, local loading of NumPy/Prophet may be blocked; the hosted app is an alternative.

## Tools and technologies

**Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Prophet · Plotly · Streamlit · GitHub**

## Limitations and future improvements

- Forecasts are **aggregated across the entire retailer**; this version does not predict for individual stores or product families.
- The dashboard allows users to change the forecast **horizon**, not provide new sales observations or retrain the model.
- Future predictions are anchored to the last available date in **2017** and should not be interpreted as present-day sales estimates.
- Forecast uncertainty grows with horizon; performance beyond the evaluated 90-day window is not established by the reported holdout metrics.
- Future extensions could include store/product-family forecasts, uploaded data with retraining, additional regressors, and a refreshed dataset.

## Author

Created as an end-to-end data science and machine-learning portfolio project.

---

**[🚀 Try the Live Dashboard](https://salestimeseriesforecasting-egqvy7htlehhkvnzf9haxc.streamlit.app/)**
