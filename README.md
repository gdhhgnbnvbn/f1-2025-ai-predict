# F1 2025 Race Winner Predictor

This project implements a machine learning model to predict Formula 1 race winners for the 2025 season using historical data from 1950 to 2024.

## Features

- Data preprocessing and feature engineering from multiple F1 datasets
- Random Forest model for race winner prediction
- Model evaluation with train/validation/test split (2022/2023/2024)
- Interactive web interface using Streamlit
- Feature importance visualization
- Model persistence for future use

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the F1 data files are in the `f1data` directory:
- circuits.csv
- constructor_results.csv
- constructor_standings.csv
- constructors.csv
- driver_standings.csv
- drivers.csv
- lap_times.csv
- pit_stops.csv
- qualifying.csv
- races.csv
- results.csv
- seasons.csv
- sprint_results.csv
- status.csv

3. Run the Streamlit application:
```bash
streamlit run f1_predictor.py
```

## Model Features

The prediction model uses the following features:
- Grid position
- Recent qualifying performance
- Points moving average (last 3 races)
- Circuit-specific performance
- Championship points and position
- Driver and constructor nationalities

## Usage

1. Open the Streamlit interface in your browser
2. Click "Train Model" to train and evaluate the model
3. View the model performance metrics and feature importance
4. The trained model will be saved for future use

## Model Performance

The model is evaluated using:
- Accuracy on training data (1950-2022)
- Accuracy on validation data (2023)
- Accuracy on test data (2024)

## Future Improvements

- Add weather data integration
- Implement pit stop strategy analysis
- Include sprint race results
- Add driver head-to-head statistics
- Implement continuous model updates during the 2025 season
