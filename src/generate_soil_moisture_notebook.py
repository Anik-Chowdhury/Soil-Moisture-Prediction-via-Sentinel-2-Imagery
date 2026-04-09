import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Cell 1: Markdown - Title and Setup
cells.append(nbf.v4.new_markdown_cell(
"""# Soil Moisture Prediction pipeline 
### 12-Band Multispectral CNN (Spatial-Spectral Architecture)

This notebook mounts Google Drive, pre-processes the 31x31 12-channel Sentinel TIF images, normalizes the data, and trains a CNN to predict `soil_moisture_depth_0.050000`."""
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)"""
))

# Cell 4: Markdown - Dataset
cells.append(nbf.v4.new_markdown_cell("## 1. Custom Dataset & Loading Logic"))

# Cell 5: Code - Custom PyTorch Dataset
cells.append(nbf.v4.new_code_cell(
"""class SoilMoistureDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get filename and ground truth
        img_name = self.df.iloc[idx]['Image_name']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load TIFF (12 channels, 31x31)
        with rasterio.open(img_path) as src:
            image = src.read() # Shape: (12, 31, 31)
            
        # Replace NaNs or extreme negative values that might exist in Sentinel NoData buffers
        image = np.nan_to_num(image, nan=0.0)
        image = np.clip(image, a_min=0, a_max=None) # Assume reflectance is >= 0
        
        # Convert to Tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        target = self.df.iloc[idx]['soil_moisture_depth_0.050000']
        target = torch.tensor([target], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target

# Band-wise Normalization Transform
class BandNormalize(object):
    \"\"\"Normalizes each of the 12 bands independently using predefined mean and std.\"\"\"
    def __init__(self, means, stds):
        self.means = torch.tensor(means).view(-1, 1, 1)
        self.stds = torch.tensor(stds).view(-1, 1, 1)
        
    def __call__(self, tensor):
        # Small epsilon to prevent division by zero
        return (tensor - self.means) / (self.stds + 1e-7)"""
))

# Cell 6: Code - Compute Stats
cells.append(nbf.v4.new_code_cell(
"""def compute_dataset_stats(df, img_dir):
    print("Computing band-wise statistics across the dataset...")
    all_pixels = []
    
    for i, row in df.iterrows():
        img_path = os.path.join(img_dir, row['Image_name'])
        if os.path.exists(img_path):
            with rasterio.open(img_path) as src:
                img = src.read()
                all_pixels.append(img)
                
    if len(all_pixels) == 0:
        raise ValueError("No images found. Please check your image directory path!")
        
    # Stack along a new dimension: (N_images, 12, 31, 31)
    all_pixels = np.stack(all_pixels)
    
    # Calculate mean and std for each band (axis 0 is images, 2 and 3 are spatial dims)
    band_means = np.mean(all_pixels, axis=(0, 2, 3))
    band_stds = np.std(all_pixels, axis=(0, 2, 3))
    
    print("Means:", band_means)
    print("Stds:", band_stds)
    return band_means, band_stds

# -----------------
# Execution Start
# -----------------
df = pd.read_csv(CSV_PATH)
print(f"Total entries in CSV: {len(df)}")

# Filter out rows where image file does not exist
available_files = set(os.listdir(IMG_DIR))
df = df[df['Image_name'].isin(available_files)].reset_index(drop=True)
print(f"Total entries after matching existing TIF files: {len(df)}")

# Train / Test split (80% Train, 20% Val)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Compute mean and std on TRAINING set only to avoid data leakage
train_means, train_stds = compute_dataset_stats(train_df, IMG_DIR)

# Augmentations for training (Random flips, preserving spatial integrity)
train_transform = transforms.Compose([
    BandNormalize(train_means, train_stds),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

val_transform = transforms.Compose([
    BandNormalize(train_means, train_stds)
])

train_dataset = SoilMoistureDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = SoilMoistureDataset(val_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)"""
))

# Cell 7: Markdown - Model
cells.append(nbf.v4.new_markdown_cell("## 2. Multi-Spectral CNN (MSS-CNN)"))

# Cell 8: Code - CNN Architecture
cells.append(nbf.v4.new_code_cell(
"""class SpatialSpectralCNN(nn.Module):
    def __init__(self, in_channels=12):
        super(SpatialSpectralCNN, self).__init__()
        
        # Extremely small receptive fields to preserve the small 31x31 spatial dimensions
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: 15x15
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # No pooling here to retain spatial info
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> Output: (128, 1, 1)
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # L2 Regularization / Dropout to prevent overfitting on small data
            nn.Linear(64, 1) # Single continuous output for moisture
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

model = SpatialSpectralCNN(in_channels=12).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay (L2)"""
))

# Cell 9: Markdown - Training Loop
cells.append(nbf.v4.new_markdown_cell("## 3. Training & Validation"))

# Cell 10: Code - Train script
cells.append(nbf.v4.new_code_cell(
"""best_val_loss = float('inf')
train_losses = []
val_losses = []

print("Starting Training...")
for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * images.size(0)
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # Calculate R2
    r2 = r2_score(np.array(all_targets), np.array(all_preds))
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | Val R2: {r2:.4f}")
    
    # Save Best Model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_soil_moisture_cnn.pth")
        
print("Training Complete. Best Validation Loss:", best_val_loss)"""
))

# Cell 11: Markdown - Metrics
cells.append(nbf.v4.new_markdown_cell("## 4. Evaluation and Visualization"))

# Cell 12: Code - Plot curves
cells.append(nbf.v4.new_code_cell(
"""# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Inference on Validation Set to plot Ground Truth vs Prediction
model.load_state_dict(torch.load("best_soil_moisture_cnn.pth"))
model.eval()

preds = []
actuals = []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds.extend(outputs.cpu().numpy().flatten())
        actuals.extend(targets.numpy().flatten())

plt.figure(figsize=(6, 6))
plt.scatter(actuals, preds, alpha=0.6)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--') # Perfect prediction line
plt.title('Actual vs Predicted Soil Moisture')
plt.xlabel('Actual Soil Moisture')
plt.ylabel('Predicted Soil Moisture')
plt.grid(True)
plt.show()"""
))


# Save the notebook
nb.cells = cells
with open(r"C:\Users\anik\Downloads\All_3_areas_cropped-20260406T074839Z-1-001\Soil_Moisture_Predictor.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
