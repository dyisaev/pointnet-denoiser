import torch
from torch import nn, optim
from nn.pointnet_denoiser import PointNet_Denoiser
from dataset.H5Dataset import DenoisingDataset
from dataset.transformations import PermuteTransform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Create the dimension permutation transform: transfrom [N,3] to [3,N]
permute = (1, 0)
transform = PermuteTransform(permute)

dataset = DenoisingDataset(
    "/diskB/data/ShapeNetCor.v2.hdf5/pcd_uniform_2048.hdf5", "car", transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
indices = torch.randperm(len(dataset)).tolist()
train_indices, test_indices = indices[:train_size], indices[train_size:]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=8, sampler=test_sampler)

# Define model
model = PointNet_Denoiser()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Specify loss function and optimizer
criterion = nn.MSELoss()  # Mean Square Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Specify number of training epochs
n_epochs = 10

# Create a SummaryWriter object
writer = SummaryWriter("runs/denoising_experiment")

for epoch in range(n_epochs):
    running_loss = 0.0

    # Training loop
    model.train()  # Set the model to training mode
    for noisy_images, clean_images in train_loader:
        # Move tensors to the same device as the model
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Forward pass: compute predicted y by passing x to the model
        outputs = model(noisy_images)

        # Compute loss
        loss = criterion(outputs, clean_images)

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss over the epoch
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs} Training Loss: {train_loss}")
    writer.add_scalar("Loss/train", train_loss, epoch)

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for evaluation
        running_loss = 0.0
        for noisy_images, clean_images in test_loader:
            # Move tensors to the same device as the model
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Forward pass: compute predicted y by passing x to the model
            outputs = model(noisy_images)

            # Compute loss
            loss = criterion(outputs, clean_images)
            running_loss += loss.item()

        # Print average loss over the test dataset
        test_loss = running_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{n_epochs} Testing Loss: {test_loss}")
        writer.add_scalar("Loss/test", test_loss, epoch)

# After all epochs are done, close the writer
writer.close()
torch.save(model, "models/pointnet_denoiser.pt")
