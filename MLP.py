"""
Multilayer Perceptron (MLP) for regression with PyTorch.

This implementation defines an MLP with one hidden layer and ReLU activation function, and trains it using stochastic gradient descent with mean squared error loss.

Data: Randomly generated 10-dimensional input and 1-dimensional output tensors using PyTorch's `randn()` function.
Author: Kyle Naddeo
"""

import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with one hidden layer.

    Args:
        input_size (int): Size of the input tensor.
        hidden_size (int, optional): Number of neurons in the hidden layer (default: 10).
        output_size (int, optional): Size of the output tensor (default: 1).
    """
    def __init__(self, input_size=10, hidden_size=10, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out) * 100.0 # Bound output between 0 and 100
        return out

def k_fold_train(model, optimizer, criterion, k, train_data, batch_size, epochs):
    """
    Train the MLP using k-fold cross-validation.

    Args:
        model (nn.Module): The MLP model to train.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (Loss): The loss function to use for training.
        k (int): The number of folds for cross-validation.
        train_data (Tensor): The training data.
        batch_size (int): The batch size for training.
        epochs (int): Number of epochs to train for.
    """
    fold_size = len(train_data) // k
    for fold in range(k):
        # Split the data into training and validation folds
        validation_data = train_data[fold * fold_size: (fold + 1) * fold_size]
        training_data = torch.cat([train_data[:fold * fold_size], train_data[(fold + 1) * fold_size:]])

        # Create data loaders for training and validation folds
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(validation_data, batch_size=batch_size)

        # Train the model for one epoch on the training data
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        # Evaluate the model on the validation data
        with torch.no_grad():
            correct = 0
            total = 0
            for data in valid_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Fold {fold + 1}, Validation Accuracy: {100 * correct / total}")

def test(model, test_loader):
    """
    Evaluate the MLP on the test data.

    Args:
        model (nn.Module): The MLP model to evaluate.
        test_loader (DataLoader): DataLoader containing the test data.
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total}")

if __name__ == "__main__":
    # Define hyperparameters
    input_size = 10
    hidden_size = 10
    output_size = 1
    batch_size = 32
    learning_rate = 0.01
    epochs = 50
    k = 5

    # Define model, loss function, and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_data = torch.randn(

