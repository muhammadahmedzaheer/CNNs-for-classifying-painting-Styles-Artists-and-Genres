import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm  # Import progress bar
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)

# Define CNN-RNN Model
class CNN_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_RNN, self).__init__()
        self.cnn = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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

# Load Model
model = CNN_RNN(num_classes).to(device)
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset path
dataset_path = os.getcwd()

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset (same as before)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Test data loader
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to check outliers based on prediction confidence
confidence_threshold = 0.5  # Set a threshold for low confidence

def is_outlier(output):
    # Apply softmax to get the probability distribution
    probs = torch.softmax(output, dim=0)  # Softmax on the correct dimension
    max_prob, predicted_class = torch.max(probs, dim=0)  # Get the class with the max probability
    return max_prob < confidence_threshold, predicted_class  # Return whether it's an outlier and predicted class

# Test Model with Progress Bar and Outlier Detection
correct, total = 0, 0
outliers = []

print("\nTesting Model:")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Progress"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images.unsqueeze(1))  # Add sequence dimension

        # Check for outliers
        for i in range(len(outputs)):
            is_outlier_flag, predicted_class = is_outlier(outputs[i])
            if is_outlier_flag:
                outliers.append((images[i], labels[i], predicted_class))  # Store the outlier samples

        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\nTest Accuracy: {test_acc:.4f}")

# Print outliers if any are found
if outliers:
    print("\nOutliers detected:")
    for i, (image, label, predicted_class) in enumerate(outliers):
        print(f"Outlier {i+1}: Predicted: {class_names[predicted_class.item()]}, Actual: {class_names[label.item()]}")
else:
    print("No outliers detected.")
