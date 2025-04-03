import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset path
dataset_path = os.getcwd()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset (ignores __pycache__)
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset: 70% Train, 15% Validation, 15% Test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Save class names
class_names = dataset.classes
with open("class_names.txt", "w") as f:
    for class_name in class_names:
        f.write(class_name + "\n")

# Define CNN-RNN Model
class CNN_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_RNN, self).__init__()
        self.cnn = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Load pretrained ResNet50
        self.cnn.fc = nn.Identity()  # Remove last FC layer
        
        self.rnn = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)  # Merge batch & sequence
        x = self.cnn(x)  # Extract features
        x = x.view(batch_size, seq_len, -1)  # Reshape back
        x, _ = self.rnn(x)  # Pass through LSTM
        x = self.fc(x[:, -1, :])  # Use last time step output
        return x

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
num_classes = len(class_names)
model = CNN_RNN(num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Resume training if checkpoint exists
start_epoch = 0
best_val_acc = 0.0

if os.path.exists("model_last.pth"):
    print("\nLoading last saved checkpoint...")
    checkpoint = torch.load("model_last.pth")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    print(f"Resuming from epoch {start_epoch}")

# Training loop
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images.unsqueeze(1))  # Add sequence dimension
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_acc = 0
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.unsqueeze(1))
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        val_acc = correct / total

    print(f"Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save latest model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc
    }
    torch.save(checkpoint, "model_last.pth")
    print(f"Checkpoint saved: model_last.pth (Epoch {epoch+1})")

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_best.pth")
        print("Best model saved as model_best.pth")

# Final Testing
model.eval()
test_acc = 0
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc="Testing Model"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images.unsqueeze(1))
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    test_acc = correct / total

print(f"Test Accuracy: {test_acc:.4f}")
