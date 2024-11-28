import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification
    using PyTorch. This model is designed for demonstration purposes
    and can be extended for more complex tasks.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Fully connected layers
        # Calculate input features for the first linear layer
        # Assuming input image size is 32x32, after two max pools (2x2 each)
        # size becomes 32/2/2 = 8. So, 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Added dropout for regularization
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Trains the given PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (cpu or cuda).
        epochs (int): Number of training epochs.
    """
    model.train()
    print(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Model training complete.")

def evaluate_model(model, test_loader, device):
    """
    Evaluates the given PyTorch model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")
    return accuracy

def load_and_prepare_cifar10(batch_size=32):
    """
    Loads the CIFAR-10 dataset, preprocesses it, and prepares it
    for training and validation with PyTorch.
    """
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=\'./data\', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=\'./data\', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print("CIFAR-10 dataset loaded and prepared.")
    return trainloader, testloader

if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    train_dl, test_dl = load_and_prepare_cifar10()
    
    # Initialize model, loss, and optimizer
    cnn_model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train model
    train_model(cnn_model, train_dl, criterion, optimizer, device, epochs=1)
    
    # Evaluate the model
    evaluate_model(cnn_model, test_dl, device)
    print("PyTorch model example completed.")
