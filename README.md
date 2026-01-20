# Flight Delay Prediction Project

A machine learning project for predicting flight delays using historical flight data and weather information.

## Project Overview

This project analyzes flight delay patterns and builds predictive models to forecast delays based on various factors including weather conditions.

## Project Structure

```
├── main.ipynb                    # Main analysis notebook
├── code.ipynb                    # Additional code notebook
├── flight_weather_pipeline.ipynb # Flight and weather data pipeline
├── *.pth                         # Trained PyTorch model files
├── Dataset/                      # Processed dataset files (2017-2025)
├── Original dataset/             # Raw flight data files (2012-2024)
├── Weather Data/                 # Weather information
├── noaa_isd_data/               # NOAA weather station data
└── venv/                        # Python virtual environment
```

## Data Sources

- **Flight Data**: Monthly flight records from 2012-2024
- **Weather Data**: NOAA ISD (Integrated Surface Database) weather station data

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Models

- `best_flight_delay_model.pth` - Best performing flight delay model
- `best_model.pth` - General best model
- `flight_delay_model_final.pth` - Final trained model

## Usage

Open and run the Jupyter notebooks in the following order:
1. `flight_weather_pipeline.ipynb` - Data preprocessing and merging
2. `main.ipynb` - Main analysis and model training

## License

This project is for educational/research purposes.
