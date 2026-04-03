import torch
import torch.nn as nn
import gc

# ── GPU Info ────────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    print("No GPU detected — using CPU. Batch size recommendation: 512")
    exit()

device = torch.device('cuda')
gpu_name        = torch.cuda.get_device_name(0)
total_vram      = torch.cuda.get_device_properties(0).total_memory
reserved        = torch.cuda.memory_reserved(0)
allocated       = torch.cuda.memory_allocated(0)
free_vram       = total_vram - reserved

print("=" * 55)
print(f"  GPU:        {gpu_name}")
print(f"  Total VRAM: {total_vram / 1024**3:.2f} GB")
print(f"  Used:       {allocated / 1024**3:.2f} GB")
print(f"  Free:       {free_vram / 1024**3:.2f} GB")
print("=" * 55)

# ── Replicate exact FlightDelayNN architecture ───────────────────────────────

INPUT_SIZE = 12   # same as train.py features_to_use

class FlightDelayNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024); self.bn1 = nn.BatchNorm1d(1024); self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512);        self.bn2 = nn.BatchNorm1d(512);  self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256);         self.bn3 = nn.BatchNorm1d(256);  self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(256, 128);         self.bn4 = nn.BatchNorm1d(128);  self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(128, 64);          self.bn5 = nn.BatchNorm1d(64);   self.dropout5 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(64, 32);           self.bn6 = nn.BatchNorm1d(32)
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


model = FlightDelayNN(INPUT_SIZE).to(device)
model.train()

total_params = sum(p.numel() for p in model.parameters())
model_vram = total_params * 4 / 1024**3   # FP32 = 4 bytes per param
print(f"\n  Model parameters: {total_params:,}")
print(f"  Model VRAM (weights): {model_vram*1024:.1f} MB")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Batch size finder ────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  FINDING OPTIMAL BATCH SIZE")
print("=" * 55)

# Test these batch sizes in order
candidates = [16384, 32768, 65536, 131072, 262144, 524288]

safe_batch   = None
max_batch    = None
results      = []

for batch_size in candidates:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        X = torch.randn(batch_size, INPUT_SIZE, device=device)
        y = torch.randint(0, 2, (batch_size,), device=device).float()

        # Forward + backward (same as real training)
        optimizer.zero_grad()
        out  = model(X).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        peak  = torch.cuda.max_memory_allocated() / 1024**3
        total_gb = total_vram / 1024**3
        pct   = (peak / total_gb) * 100

        results.append((batch_size, peak, pct))
        print(f"  Batch {batch_size:>6,}  Peak VRAM: {peak:.2f} GB  ({pct:.1f}%)", end="")

        if pct < 75:
            safe_batch = batch_size
            print("  [safe - can go higher]")
        elif pct < 90:
            safe_batch = batch_size
            max_batch  = batch_size
            print("  [TARGET - use this]")
        elif pct < 98:
            max_batch  = batch_size
            print("  [tight - risky]")
        else:
            print("  [OOM risk - too much]")

        del X, y, out, loss

    except torch.cuda.OutOfMemoryError:
        print(f"  Batch {batch_size:>6,}  OUT OF MEMORY")
        torch.cuda.empty_cache()
        break

# ── Recommendation ───────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  RECOMMENDATION")
print("=" * 55)

recommended = safe_batch if safe_batch else 512

print(f"\n  Recommended BATCH_SIZE = {recommended}")
print(f"\n  In train.py, set:")
print(f"\n      BATCH_SIZE = {recommended}")
print()

# Safety margin explanation
if results:
    for bs, peak, pct in results:
        if bs == recommended:
            headroom = (total_vram / 1024**3) - peak
            print(f"  This uses {peak:.2f} GB of your {total_vram/1024**3:.1f} GB VRAM")
            print(f"  Headroom: {headroom:.2f} GB (for BatchNorm stats, optimizer state, etc.)")

print("=" * 55)
