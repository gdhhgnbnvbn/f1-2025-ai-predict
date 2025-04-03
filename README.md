# F1 2025 Race Winner Predictor 🏎️

An AI-powered Formula 1 prediction system that uses machine learning to forecast race winners and championship outcomes for the 2025 F1 season. Built with Python, scikit-learn, and Streamlit.

## Features

- **Single Race Predictions**: Predict the outcome of any 2025 F1 race with detailed win probabilities
- **Championship Simulation**: Simulate the entire 2025 season and get final standings predictions
- **Real-time Updates**: Incorporates actual 2025 race results into predictions
- **Interactive UI**: Modify grid positions to see how qualifying affects race outcomes
- **Comprehensive Stats**: View driver and constructor championship predictions

## Current 2025 Data

- Includes real results from the first two races:
  - Australian GP (Melbourne): Won by Lando Norris (McLaren)
  - Chinese GP (Shanghai): Won by Oscar Piastri (McLaren)
- Updated 2025 driver lineup with all mid-season changes
- Complete 24-race calendar with correct dates and venues

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/f1-2025-winner.git
cd f1-2025-winner
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
python -m streamlit run f1_predictor.py
```

## Usage

### Race Predictions

1. Go to the "2025 Predictions" tab
2. Select a race from the dropdown menu
3. (Optional) Modify grid positions
4. Click "Predict Race Results"
5. View detailed predictions including win probabilities

### Championship Predictions

1. Navigate to the "Championship Prediction" tab
2. Click "Simulate Remaining Races"
3. View:
   - Final Driver's Championship standings
   - Constructor's Championship standings
   - Predicted podiums for all remaining races
   - Points distribution visualization

## How It Works

The prediction system uses a Random Forest model trained on historical F1 data from 1950-2024, considering factors such as:

- Qualifying position
- Recent driver performance
- Team performance
- Circuit-specific statistics
- Championship position
- Historical results

## Data Sources

- Historical F1 data (1950-2024)
- Real-time 2025 season results
- Current driver and constructor information
- Official 2025 F1 calendar

## Contributing

Feel free to open issues or submit pull requests with improvements.


## Acknowledgments

- Formula 1 for providing historical data
- Streamlit for the amazing web framework
- The F1 community for inspiration and feedback

## Disclaimer

This is a predictive model for entertainment purposes. Actual race results may vary significantly from predictions.
