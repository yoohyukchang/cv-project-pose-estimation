from scripts.coco_data_loader import get_coco_dataloader, HeatmapGenerator
from models.AlphaPose import AlphaPoseModel
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, random_split

def main():
    num_keypoints = 17
    output_size = (64, 48)  # (height, width)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    target_transform = HeatmapGenerator(output_size=output_size, num_keypoints=num_keypoints)

    # Get the full training DataLoader
    full_train_loader = get_coco_dataloader(batch_size=16, phase="train", transform=transform, target_transform=target_transform)

    # Define the split ratio and compute the sizes for train and validation subsets
    val_split_ratio = 0.2
    dataset_size = len(full_train_loader.dataset)
    val_size = int(dataset_size * val_split_ratio)
    train_size = dataset_size - val_size

    # Split the dataset into training and validation subsets
    train_subset, val_subset = random_split(full_train_loader.dataset, [train_size, val_size])

    # Create DataLoaders for the train and validation subsets
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    # Determine the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model and move it to the appropriate device
    model = AlphaPoseModel(num_keypoints=num_keypoints).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s')

    # Save the trained model
    torch.save(model.state_dict(), 'pose_estimation_model.pth')


if __name__ == "__main__":
    main()
