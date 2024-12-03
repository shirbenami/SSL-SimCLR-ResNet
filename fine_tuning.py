import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from model.resnet import build_resnet50
from trainers.train import train_model
from trainers.validate import validate_model
from trainers.test import test_model

# Settings
batch_size = 32
learning_rate = 0.001
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl_weights_path = "output/models/simclr_model2.pth"  # Path to SSL weights

# Data transformations
transform = Compose([
    RandomResizedCrop(96),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset_full = STL10(root='./data', split='train', download=True, transform=transform)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
test_dataset = STL10(root='./data', split='test', download=True, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = [str(i) for i in range(10)]

print(f"Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

# Build model
model = build_resnet50()
model.to(device)

# Load SSL weights
print("Loading SSL weights...")
ssl_weights = torch.load(ssl_weights_path, map_location=device)
model.load_state_dict(ssl_weights, strict=False)  # strict=False to allow for slight differences
print("SSL weights loaded successfully.")

# Freeze ResNet layers
for param in model.parameters():  # Assuming model[0] is the ResNet encoder
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Training and validation loops
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Save results as graphs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Loss graph
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Accuracy graph
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("./output/logs/fine_tuning_classification_graphs.png")
plt.show()

# Test model
test_accuracy = test_model(model, test_loader, device, class_names)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the fine-tuned model
torch.save(model.state_dict(), "./output/models/fine_tuned_classification_model.pth")
print("Fine-tuned classification model saved successfully!")
