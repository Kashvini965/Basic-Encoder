import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1] range automatically
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get a sample to check dimensions
sample_image, _ = next(iter(train_loader))
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Image shape: {sample_image.shape}")
print(f"Image value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")

# Flatten all data for reconstruction
x_train_flat = sample_image.view(sample_image.size(0), -1).cpu().numpy()
print(f"Flattened image shape: {x_train_flat.shape}")

# Define Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Initialize model
input_dim = 28 * 28
latent_dim = 32
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

# Display model architecture
print("\nAutoencoder Architecture:")
print("=" * 60)
print(f"Input dimension: {input_dim}")
print(f"Latent dimension: {latent_dim}")
print("=" * 60)
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nLoss Function: Mean Squared Error (MSE)")
print("Optimizer: Adam (learning_rate=0.001)")
print("\nModel setup completed!")

# Training functions
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten images
        data = data.view(data.size(0), -1).to(device)
        
        # Forward pass
        reconstructed, latent = model(data)
        loss = criterion(reconstructed, data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            reconstructed, latent = model(data)
            loss = criterion(reconstructed, data)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# Training loop
num_epochs = 30
train_losses = []
val_losses = []

print(f"\nTraining the autoencoder for {num_epochs} epochs...")
print("=" * 70)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

print("=" * 70)
print("Training completed!")

# Evaluate on test set
print("\nEvaluating reconstruction performance on test set...")
print("=" * 70)

model.eval()
all_reconstructed = []
all_original = []
per_sample_errors = []

with torch.no_grad():
    for data, _ in test_loader:
        data = data.view(data.size(0), -1).to(device)
        reconstructed, _ = model(data)
        
        all_original.append(data.cpu().numpy())
        all_reconstructed.append(reconstructed.cpu().numpy())
        
        # Calculate per-sample errors
        sample_errors = np.mean((data.cpu().numpy() - reconstructed.cpu().numpy()) ** 2, axis=1)
        per_sample_errors.extend(sample_errors)

# Concatenate all batches
x_test_flat = np.concatenate(all_original, axis=0)
x_test_pred = np.concatenate(all_reconstructed, axis=0)
per_sample_mse = np.array(per_sample_errors)

# Calculate metrics
mse = mean_squared_error(x_test_flat, x_test_pred)
mae = mean_absolute_error(x_test_flat, x_test_pred)
rmse = np.sqrt(mse)

# Print evaluation results
print(f"\nReconstruction Error Metrics (Test Set):")
print(f"  Mean Squared Error (MSE):        {mse:.6f}")
print(f"  Root Mean Squared Error (RMSE):  {rmse:.6f}")
print(f"  Mean Absolute Error (MAE):       {mae:.6f}")
print(f"\nPer-Sample MSE Statistics:")
print(f"  Mean:   {per_sample_mse.mean():.6f}")
print(f"  Std:    {per_sample_mse.std():.6f}")
print(f"  Min:    {per_sample_mse.min():.6f}")
print(f"  Max:    {per_sample_mse.max():.6f}")
print(f"  Median: {np.median(per_sample_mse):.6f}")
print("=" * 70)

print("\n✅ Autoencoder training and evaluation completed successfully!")
print("\nResults Summary:")
print(f"  - Final Training Loss: {train_losses[-1]:.6f}")
print(f"  - Final Validation Loss: {val_losses[-1]:.6f}")
print(f"  - Test MSE: {mse:.6f}")
print(f"  - Test MAE: {mae:.6f}")
