import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Cell 1: Markdown - Title and Setup
cells.append(nbf.v4.new_markdown_cell(
"""# Temporal Soil Moisture Prediction pipeline 
### CNN-LSTM Hybrid Architecture (Spatial-Spectral + Temporal)

This notebook mounts Google Drive, sequences historical 31x31 12-channel Sentinel TIF images natively across time, and trains a CNN-LSTM to predict `soil_moisture_depth_0.050000`."""
))

# Cell 2: Code - Setup Environment
cells.append(nbf.v4.new_code_cell(
"""!pip install rasterio earthpy

import os
import glob
import pandas as pd
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""
))

# Cell 3: Code - Hyperparameters and Paths
cells.append(nbf.v4.new_code_cell(
"""# --- USER CONFIGURATION ---
# Change these paths to exactly where your folders are located in your Google Drive!
BASE_DIR = '/content/drive/MyDrive/All_3_areas_cropped-20260406T074839Z-1-001'
CSV_PATH = os.path.join(BASE_DIR, 'Combined_Area1_Area2_Area3_Area5_Area6_with_tif.csv')
IMG_DIR = os.path.join(BASE_DIR, 'All_3_areas_cropped')

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
SEQ_LEN = 5 # Number of historical timesteps to use (including current day)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)"""
))

# Cell 4: Markdown - Dataset
cells.append(nbf.v4.new_markdown_cell("## 1. Sequence Modeling Dataset"))

# Cell 5: Code - Custom PyTorch Dataset
cells.append(nbf.v4.new_code_cell(
"""class SoilMoistureSequenceDataset(Dataset):
    def __init__(self, sequences, img_dir, transform=None):
        \"\"\"
        sequences: List of pandas DataFrames, where each DataFrame represents a sequence 
                   of length SEQ_LEN for a specific area, ordered by chronological date.
        \"\"\"
        self.sequences = sequences
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_df = self.sequences[idx]
        
        sequence_images = []
        for _, row in seq_df.iterrows():
            img_name = row['Image_name']
            img_path = os.path.join(self.img_dir, img_name)
            
            with rasterio.open(img_path) as src:
                image = src.read() # (12, 31, 31)
                
            image = np.nan_to_num(image, nan=0.0)
            image = np.clip(image, a_min=0, a_max=None)
            image = torch.tensor(image, dtype=torch.float32)
            
            if self.transform:
                image = self.transform(image)
                
            sequence_images.append(image)
            
        # Stack images into (seq_len, 12, 31, 31)
        sequence_tensor = torch.stack(sequence_images) 
        
        # The target is the soil moisture of the LAST day in the sequence
        target_val = seq_df.iloc[-1]['soil_moisture_depth_0.050000']
        target = torch.tensor([target_val], dtype=torch.float32)

        return sequence_tensor, target

class BandNormalize(object):
    \"\"\"Normalizes each of the 12 bands independently across the dataset.\"\"\"
    def __init__(self, means, stds):
        self.means = torch.tensor(means).view(-1, 1, 1)
        self.stds = torch.tensor(stds).view(-1, 1, 1)
        
    def __call__(self, tensor):
        return (tensor - self.means) / (self.stds + 1e-7)"""
))

# Cell 6: Code - Sequence grouping and processing
cells.append(nbf.v4.new_code_cell(
"""def compute_dataset_stats(sequences, img_dir):
    print("Computing band-wise statistics across the training set...")
    all_pixels = []
    
    for seq_df in sequences:
        for _, row in seq_df.iterrows():
            img_path = os.path.join(img_dir, row['Image_name'])
            with rasterio.open(img_path) as src:
                img = src.read()
                all_pixels.append(img)
                
    all_pixels = np.stack(all_pixels)
    band_means = np.mean(all_pixels, axis=(0, 2, 3))
    band_stds = np.std(all_pixels, axis=(0, 2, 3))
    
    return band_means, band_stds

def create_sequences(df, seq_len):
    sequences = []
    # Group by Geographic Area
    grouped = df.groupby('Area')
    for area, group in grouped:
        # Sort chronologically
        group = group.sort_values('date_time').reset_index(drop=True)
        if len(group) < seq_len:
            continue
            
        # Sliding window approach
        for i in range(len(group) - seq_len + 1):
            seq = group.iloc[i : i+seq_len]
            sequences.append(seq)
    return sequences

# -----------------
# Execution Start
# -----------------
df = pd.read_csv(CSV_PATH)
df['date_time'] = pd.to_datetime(df['date_time']) # Ensure date sorting is correct

# Filter out missing image files immediately
available_files = set(os.listdir(IMG_DIR))
df = df[df['Image_name'].isin(available_files)].reset_index(drop=True)

# Build Sequences per Area
all_sequences = create_sequences(df, SEQ_LEN)
print(f"Total temporal sequences formed (length {SEQ_LEN}): {len(all_sequences)}")

# Train / Test split (80% Train, 20% Val)
# Splitting on sequence indexes
train_seqs, val_seqs = train_test_split(all_sequences, test_size=0.2, random_state=42)

# Compute mean and std on TRAINING sequences only
train_means, train_stds = compute_dataset_stats(train_seqs, IMG_DIR)

# Transforms 
train_transform = transforms.Compose([
    BandNormalize(train_means, train_stds)
    # Note: Spatial augmentations (like flip) on temporal sequences must be applied 
    # strictly identically across ALL timesteps in a sequence. To avoid bugs, we omit 
    # naive random flips here. 
])

val_transform = transforms.Compose([
    BandNormalize(train_means, train_stds)
])

train_dataset = SoilMoistureSequenceDataset(train_seqs, IMG_DIR, transform=train_transform)
val_dataset = SoilMoistureSequenceDataset(val_seqs, IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)"""
))

# Cell 7: Markdown - Model
cells.append(nbf.v4.new_markdown_cell("## 2. Hybrid CNN-LSTM Model"))

# Cell 8: Code - CNN-LSTM Architecture
cells.append(nbf.v4.new_code_cell(
"""class CNNLSTM(nn.Module):
    def __init__(self, in_channels=12, cnn_embed_dim=128, lstm_hidden=64):
        super(CNNLSTM, self).__init__()
        
        # 1. Feature Extractor (Spatial-Spectral CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> 15x15
            
            nn.Conv2d(64, cnn_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_embed_dim),
            nn.ReLU(),
            
            nn.Conv2d(cnn_embed_dim, cnn_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average -> (cnn_embed_dim, 1, 1)
        )
        
        # 2. Temporal Modeling (LSTM)
        # batch_first=True makes IO tensors shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=cnn_embed_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        
        # 3. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x is originally shaped: (batch_size, seq_len, channels, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Collapse batch and sequence dimensions to process all images through CNN
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        
        cnn_out = self.cnn(x_reshaped) # Output shape: (batch_size*seq_len, cnn_embed_dim, 1, 1)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) # Reshape for LSTM: (batch, seq, embed_dim)
        
        # Pass sequence features through LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out) 
        
        # Extract the hidden state from the last timestep
        last_step_out = lstm_out[:, -1, :] # Shape: (batch, lstm_hidden)
        
        # Regress to a continuous moisture output
        return self.regressor(last_step_out)

model = CNNLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)"""
))

# Cell 9: Markdown - Training Loop
cells.append(nbf.v4.new_markdown_cell("## 3. Training & Validation"))

# Cell 10: Code - Train script
cells.append(nbf.v4.new_code_cell(
"""best_val_loss = float('inf')
train_losses = []
val_losses = []

print("Starting Setup-Time and Training Sequence...")
for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    for seq_images, targets in train_loader:
        seq_images, targets = seq_images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq_images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * seq_images.size(0)
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for seq_images, targets in val_loader:
            seq_images, targets = seq_images.to(device), targets.to(device)
            outputs = model(seq_images)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * seq_images.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    r2 = r2_score(np.array(all_targets), np.array(all_preds))
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | Val R2: {r2:.4f}")
    
    # Save Best Model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_soil_moisture_hybrid.pth")
        
print("Training Complete. Best Validation Loss:", best_val_loss)"""
))

# Cell 11: Markdown - Metrics
cells.append(nbf.v4.new_markdown_cell("## 4. Evaluation Curves"))

# Cell 12: Code - Plot curves
cells.append(nbf.v4.new_code_cell(
"""# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss (CNN-LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Inference on Validation Set to plot Ground Truth vs Prediction
model.load_state_dict(torch.load("best_soil_moisture_hybrid.pth"))
model.eval()

preds = []
actuals = []

with torch.no_grad():
    for seq_images, targets in val_loader:
        seq_images = seq_images.to(device)
        outputs = model(seq_images)
        preds.extend(outputs.cpu().numpy().flatten())
        actuals.extend(targets.numpy().flatten())

plt.figure(figsize=(6, 6))
plt.scatter(actuals, preds, alpha=0.6)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--')
plt.title('Actual vs Predicted Soil Moisture (CNN-LSTM)')
plt.xlabel('Ground Truth Soil Moisture')
plt.ylabel('Predicted Soil Moisture')
plt.grid(True)
plt.show()"""
))

# Save the notebook
nb.cells = cells
save_path = os.path.join(r"C:\\Users\\anik\\Downloads\\All_3_areas_cropped-20260406T074839Z-1-001", "Soil_Moisture_CNNLSTM.ipynb")
with open(save_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Hybrid CNN-LSTM Notebook successfully created at: {save_path}")
