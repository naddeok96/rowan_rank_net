"""
MLP Regression with PyTorch and WandB Hyperparameter Sweeps

This code demonstrates how to implement a multilayer perceptron (MLP) for regression with PyTorch, and how to use WandB's hyperparameter sweep functionality to find the best hyperparameters for the model. The MLP has one hidden layer and ReLU activation function, and is trained using stochastic gradient descent with mean squared error loss. 

The code includes a YAML file for configuring the hyperparameter sweep, and uses the Bayesian search method with a quantized logarithmic distribution for the learning rate.

Author: Kyle Naddeo
Last updated: March 12, 2023

By logging the model's performance and hyperparameters to WandB, this code enables easy tracking and comparison of different hyperparameter configurations and allows for seamless collaboration with teammates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml


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
        train_data (Tensor): The training data. [n_samples, output_size + n_features]
        batch_size (int): The batch size for training.
        epochs (int): Number of epochs to train for.
    """
    fold_size = len(train_data) // k
    validation_accuracies = []  # List to store validation accuracies for each fold
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
                inputs, labels = data[:,1:], data[:,0].unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Log training loss to wandb
            wandb.log({'training_loss': running_loss / len(train_loader)})

        # Evaluate the model on the validation data
        with torch.no_grad():
            correct = 0
            total = 0
            for data in valid_loader:
                inputs, labels = data[:,1:], data[:,0].unsqueeze(1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Log validation accuracy to wandb
            validation_accuracy = 100 * correct / total
            wandb.log({'validation_accuracy': validation_accuracy})
            validation_accuracies.append(validation_accuracy)

    # Calculate the mean validation accuracy
    mean_validation_accuracy = sum(validation_accuracies) / len(validation_accuracies)
    wandb.log({'mean_validation_accuracy': mean_validation_accuracy})

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
            inputs, labels = data[:,1:], data[:,0].unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Log test accuracy to wandb
        wandb.log({'test_accuracy': 100 * correct / total})

if __name__ == "__main__":

    # Load the sweep configuration from YAML file
    with open('sweep.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, entity="naddeok", project="Rowan Rank Net")

    # Define training function
    def train():
        run = wandb.init()
        
        # Define hyperparameters
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        hidden_size = wandb.config.hidden_size
        epochs = wandb.config.epochs
        k = 5
        
        # Load data
        train_data = torch.randn(200, 11)

        # Define model, loss function, and optimizer
        model = MLP(10, hidden_size, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate model
        k_fold_train(model, optimizer, criterion, k, train_data, batch_size, epochs)
        test(model, DataLoader(train_data, batch_size=batch_size))

    # Run sweep
    wandb.agent(sweep_id, train)
    

