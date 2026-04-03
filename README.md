# Flight Delay Prediction

A deep learning pipeline that predicts whether a US domestic flight will be delayed by 15+ minutes, trained on 1.4 million BTS flight records using a GPU-accelerated PyTorch neural network.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **84.48%** |
| Delay Rate in Data | 16.63% |
| Training Samples | 1,422,636 |
| Model Parameters | 716,737 |
| Training Device | NVIDIA GTX 1650 (GPU) |

## Project Structure

```
Flight_Delay_Prediction/
├── preprocess.py          # ETL pipeline — cleans raw BTS CSVs, engineers features
├── train.py               # PyTorch training — model definition, GPU training loop, evaluation
├── analyze_delays.py      # EDA — delay patterns by hour, day, airline, cause
├── gpu_check.py           # GPU memory profiler — finds optimal batch size
├── visualize_network.py   # Neural network architecture diagram
└── analysis/
    └── delay_analysis_report.md
```

## Model Architecture

7-layer fully connected neural network (FlightDelayNN):

```
Input (12 features)
  → Linear(12 → 1024) + BatchNorm + ReLU + Dropout(0.30)
  → Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.30)
  → Linear(512 → 256)  + BatchNorm + ReLU + Dropout(0.25)
  → Linear(256 → 128)  + BatchNorm + ReLU + Dropout(0.20)
  → Linear(128 → 64)   + BatchNorm + ReLU + Dropout(0.20)
  → Linear(64 → 32)    + BatchNorm + ReLU
  → Linear(32 → 1)     + Sigmoid
Output: delay probability
```

**Loss:** Binary Cross Entropy | **Optimizer:** Adam (lr=0.001) | **Scheduler:** ReduceLROnPlateau

## Features Used

| Feature | Description |
|---------|-------------|
| MONTH | Month of year |
| DAY_OF_WEEK | Day of week |
| DEP_TIME_CAT | Departure time category (morning/afternoon/evening/night) |
| CRS_DEP_TIME | Scheduled departure time |
| CRS_ARR_TIME | Scheduled arrival time |
| DISTANCE | Flight distance (miles) |
| TAXI_OUT | Taxi-out time (minutes) |
| TEMPERATURE | Origin airport temperature |
| PRECIPITATION_PROB | Precipitation probability |
| WIND_SPEED | Wind speed at origin |
| VISIBILITY | Visibility at origin |
| WEATHER_BAD | Engineered binary weather flag |

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install torch pandas scikit-learn matplotlib seaborn numpy

# 3. Place raw BTS flight CSVs in: Original dataset/
```

## Usage

```bash
# Step 1 — Preprocess raw BTS data (run once)
python preprocess.py

# Step 2 — Exploratory data analysis
python analyze_delays.py

# Step 3 — Find optimal GPU batch size
python gpu_check.py

# Step 4 — Train the model
python train.py
```

## Data Source

[Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/DL_SelectFields.aspx) — On-Time Performance dataset, domestic US flights 2012–2024.

## Key Design Decisions

- **Chunked I/O + parallel preprocessing** — 157 monthly CSVs processed with `ProcessPoolExecutor(4 workers)` to handle 81M raw rows without running out of RAM
- **GPU batch size 131,072** — fills ~3.2GB of 4GB VRAM; tensors pinned to GPU at dataset init to avoid repeated CPU→GPU transfers
- **BatchNorm on every layer** — stabilizes training on tabular data with mixed feature scales
- **Class imbalance** — dataset is 83% on-time / 17% delayed; model achieves high accuracy but low recall on delays (known limitation; SMOTE or class weights would improve this)
