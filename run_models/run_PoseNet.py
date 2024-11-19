from scripts.coco_data_loader import get_coco_dataloader, HeatmapGenerator
from models.PoseNet import PoseNet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms
import torchvision.transforms.functional as F

def train_posenet(num_epochs=10, batch_size=16, learning_rate=1e-4, train_samples=1000, val_samples=200):
    # Dataset transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = HeatmapGenerator(output_size=(64, 48), num_keypoints=17)
    
    print(f"Loading datasets (train: {train_samples} samples, val: {val_samples} samples)...")
    train_loader = get_coco_dataloader(
        batch_size=batch_size,
        phase="train",
        transform=transform,
        target_transform=target_transform,
        max_samples=train_samples
    )
    
    val_loader = get_coco_dataloader(
        batch_size=batch_size,
        phase="val",
        transform=transform,
        target_transform=target_transform,
        max_samples=val_samples
    )
    
    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseNet(num_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Time: {epoch_time:.2f}s')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_posenet_model.pth')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return model, history

def load_best_model(model_path='best_posenet_model.pth'):
    """
    Load the best saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseNet(num_keypoints=17).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    return model

def main():
    # Train the model
    model, history = train_posenet(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        train_samples=1000,
        val_samples=200
    )

    best_model = load_best_model('best_posenet_model.pth')


if __name__ == "__main__":
    main()
