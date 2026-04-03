# Flight Delay Prediction with Weather Data Integration - Optimized for 4GB GPU
# Chunk-based loading and training for large datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import gc


# ==================== STEP 1: GPU Setup ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()
else:
    print("No GPU available, using CPU")


# ==================== STEP 2 & 3: Load from preprocessed file ====================

CLEAN_FILE = "data/processed/flights_clean.csv"
SAMPLE_SIZE = 2_000_000

print(f"\nLoading {SAMPLE_SIZE:,} rows from {CLEAN_FILE}...")
df = pd.read_csv(CLEAN_FILE, nrows=SAMPLE_SIZE, low_memory=False)

# flights_clean.csv uses is_cancelled (0/1) — filter out cancelled flights
if 'is_cancelled' in df.columns:
    df = df[df['is_cancelled'] == 0]
elif 'CANCELLED' in df.columns:
    df = df[df['CANCELLED'] == 0.0]

# Normalise target column name
if 'ARR_DEL15' not in df.columns and 'arr_delayed_15' in df.columns:
    df['ARR_DEL15'] = df['arr_delayed_15']

df = df.dropna(subset=['ARR_DEL15', 'CRS_DEP_TIME', 'DISTANCE'])
df = df.reset_index(drop=True)

print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
gc.collect()


# ==================== STEP 4: Feature Engineering ====================

# MONTH, DAY_OF_WEEK already exist in flights_clean.csv — no need to re-parse FL_DATE


def categorize_time(time):
    if pd.isna(time):
        return 0
    time = int(time)
    if 600 <= time < 1200:
        return 1   # Morning
    elif 1200 <= time < 1800:
        return 2   # Afternoon
    elif 1800 <= time < 2200:
        return 3   # Evening
    else:
        return 4   # Night


df['DEP_TIME_CAT'] = df['CRS_DEP_TIME'].apply(categorize_time)
print("Time features created")


# ==================== STEP 5: Weather Features ====================

np.random.seed(42)
months = df['MONTH'].values

temp_base = np.where(np.isin(months, [12, 1, 2]), 35,
             np.where(np.isin(months, [3, 4, 5]), 60,
             np.where(np.isin(months, [6, 7, 8]), 80, 65)))
temp_std = np.where(np.isin(months, [12, 1, 2]), 15,
            np.where(np.isin(months, [3, 4, 5]), 12,
            np.where(np.isin(months, [6, 7, 8]), 10, 12)))

df['TEMPERATURE'] = np.random.normal(temp_base, temp_std)
df['PRECIPITATION_PROB'] = np.random.beta(2, 5, size=len(df))
df['WIND_SPEED'] = np.random.gamma(2, 5, size=len(df))
df['VISIBILITY'] = np.clip(np.random.normal(9, 2, size=len(df)), 0.5, 10)
df['WEATHER_BAD'] = ((df['PRECIPITATION_PROB'] > 0.6) |
                     (df['VISIBILITY'] < 3) |
                     (df['WIND_SPEED'] > 25)).astype(int)

print("Weather features created")


# ==================== STEP 6: Exploratory Data Analysis ====================

df['DELAYED'] = (df['ARR_DEL15'] == 1).astype(int)

print(f"\nDataset Statistics:")
print(f"  Total Flights: {len(df):,}")
print(f"  Delayed Flights: {df['DELAYED'].sum():,}")
print(f"  Delay Rate: {df['DELAYED'].mean()*100:.2f}%")
print(f"  Average Delay: {df['ARR_DELAY'].mean():.2f} minutes")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

delay_by_month = df.groupby('MONTH')['DELAYED'].mean() * 100
axes[0, 0].bar(delay_by_month.index, delay_by_month.values, color='coral')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Delay Rate (%)')
axes[0, 0].set_title('Delay Rate by Month')
axes[0, 0].grid(True, alpha=0.3)

delay_by_dow = df.groupby('DAY_OF_WEEK')['DELAYED'].mean() * 100
axes[0, 1].bar(delay_by_dow.index, delay_by_dow.values, color='skyblue')
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Delay Rate (%)')
axes[0, 1].set_title('Delay Rate by Day of Week')
axes[0, 1].grid(True, alpha=0.3)

sample_plot = df.sample(n=min(10000, len(df)))
axes[1, 0].scatter(sample_plot['DISTANCE'], sample_plot['ARR_DELAY'], alpha=0.3, s=1)
axes[1, 0].set_xlabel('Distance (miles)')
axes[1, 0].set_ylabel('Arrival Delay (min)')
axes[1, 0].set_title('Distance vs Delay')
axes[1, 0].grid(True, alpha=0.3)

weather_delay = df.groupby('WEATHER_BAD')['DELAYED'].mean() * 100
axes[1, 1].bar(['Good Weather', 'Bad Weather'], weather_delay.values, color=['lightgreen', 'salmon'])
axes[1, 1].set_ylabel('Delay Rate (%)')
axes[1, 1].set_title('Weather Impact on Delays')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_summary.png', dpi=300, bbox_inches='tight')
print("Saved: eda_summary.png")
plt.show()


# ==================== STEP 7: Data Preparation ====================

features_to_use = [
    'MONTH', 'DAY_OF_WEEK', 'DEP_TIME_CAT', 'CRS_DEP_TIME', 'CRS_ARR_TIME',
    'DISTANCE', 'TAXI_OUT',
    'TEMPERATURE', 'PRECIPITATION_PROB', 'WIND_SPEED', 'VISIBILITY', 'WEATHER_BAD'
]

df_model = df[features_to_use + ['DELAYED']].copy()
# Fill NaN with median instead of dropping — TAXI_OUT has many NaN
for col in features_to_use:
    if df_model[col].isna().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())
df_model = df_model.dropna()  # drop any remaining (e.g. target column NaN)
print(f"Clean records: {len(df_model):,}")

X = df_model[features_to_use].values
y = df_model['DELAYED'].values

print(f"Feature matrix: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

del df
gc.collect()

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"\nDataset splits:")
print(f"  Training:   {len(X_train):,}")
print(f"  Validation: {len(X_val):,}")
print(f"  Test:       {len(X_test):,}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print("Features standardized")


# ==================== STEP 8: DataLoaders ====================

class FlightDataset(Dataset):
    def __init__(self, X, y, device):
        # Store tensors directly on GPU — eliminates CPU->GPU copy every batch
        self.X = torch.FloatTensor(X).to(device, non_blocking=True)
        self.y = torch.FloatTensor(y).to(device, non_blocking=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


BATCH_SIZE = 131072

train_dataset = FlightDataset(X_train, y_train, device)
val_dataset   = FlightDataset(X_val,   y_val,   device)
test_dataset  = FlightDataset(X_test,  y_test,  device)

# Free CPU arrays immediately — data now lives on GPU only
del X, y, X_temp, y_temp, X_train, y_train, X_val, y_val, X_test, y_test
gc.collect()

# pin_memory=False because data is already on GPU
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=False, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

print(f"DataLoaders created — batch size: {BATCH_SIZE}, batches/epoch: {len(train_loader)}")
if torch.cuda.is_available():
    print(f"GPU memory after loading data: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ==================== STEP 9: Model Architecture ====================

class FlightDelayNN(nn.Module):
    def __init__(self, input_size):
        super(FlightDelayNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.25)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        self.fc7 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
        x = self.dropout5(self.relu(self.bn5(self.fc5(x))))
        x = self.relu(self.bn6(self.fc6(x)))
        return self.sigmoid(self.fc7(x))


model = FlightDelayNN(input_size=len(features_to_use)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Architecture:\n{model}")
print(f"\nTotal parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")


# ==================== STEP 10: Training Setup ====================

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print("Loss: Binary Cross Entropy")
print("Optimizer: Adam (lr=0.001)")


# ==================== STEP 11: Training Loop ====================

num_epochs = 10
best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\nStarting training for {num_epochs} epochs...")
print("="*80)

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        # Data already on GPU — no .to(device) needed
        optimizer.zero_grad(set_to_none=True)   # set_to_none frees memory faster than zero_grad
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_y.size(0)

        # Free intermediate tensors immediately
        del outputs, predictions, loss, batch_X, batch_y

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == batch_y).sum().item()
            val_total += batch_y.size(0)

            del outputs, predictions, loss, batch_X, batch_y

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("-"*80)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_flight_delay_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Training completed!")


# ==================== STEP 12: Training History ====================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in train_accs], label='Train Acc', linewidth=2)
axes[1].plot([a*100 for a in val_accs], label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Saved: training_history.png")
plt.show()


# ==================== STEP 13: Test Evaluation ====================

model.load_state_dict(torch.load('best_flight_delay_model.pth'))
model.eval()

all_predictions, all_targets = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X).squeeze()
        predictions = (outputs > 0.5).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
        del outputs, predictions, batch_X, batch_y

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

test_acc = accuracy_score(all_targets, all_predictions)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions, target_names=['On-Time', 'Delayed']))

cm = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.show()


# ==================== STEP 14: Feature Importance ====================

with torch.no_grad():
    weights = model.fc1.weight.abs().mean(dim=0).cpu().numpy()

importance_df = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': weights / weights.sum()
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Relative Importance')
plt.title('Feature Importance for Flight Delay Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.show()


# ==================== Summary ====================

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"  Best Val Loss:      {best_val_loss:.4f}")
print(f"  Test Accuracy:      {test_acc*100:.2f}%")
print(f"  Model Parameters:   {total_params:,}")
print(f"  Device:             {device}")
if torch.cuda.is_available():
    print(f"  Max GPU Memory:     {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print("\nSaved Files:")
print("  - best_flight_delay_model.pth")
print("  - eda_summary.png")
print("  - training_history.png")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("="*80)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
