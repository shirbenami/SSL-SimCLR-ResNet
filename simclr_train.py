import torch
from model.ssl_model import build_classifier_STL10
from trainers_ssl.train import train_model
from trainers_ssl.validate import validate_model
# Define number of epochs
num_epochs = 15

# Load the model, dataloaders, loss function, and optimizer
train_loader, val_loader, model, criterion, optimizer = build_classifier_STL10(lr=0.001)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    # Training step
    train_loss = train_model(model, train_loader, criterion, optimizer, device)

    # Validation step
    val_loss = validate_model(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    model.train()
    total_loss = 0.0

# Save results as graphs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Loss graph
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("./output/logs/ssl_losses_graphs.png")
plt.show()


# Save the trained model
torch.save(model.state_dict(), "simclr_model.pth")
print("Model saved successfully!")
