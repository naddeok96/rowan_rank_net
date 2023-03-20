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
from data_processing import process_data

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

def k_fold_train(model, optimizer, criterion, k, train_data, batch_size, epochs, use_gpu):
    """
    Train the MLP using k-fold cross-validation for regression.

    Args:
        model (nn.Module): The MLP model to train.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (Loss): The loss function to use for training.
        k (int): The number of folds for cross-validation.
        train_data (Tensor): The training data. [n_samples, output_size + n_features]
        batch_size (int): The batch size for training.
        epochs (int): Number of epochs to train for.
        use_gpu (bool): Whether to use a GPU or not.
    """
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    fold_size = len(train_data) // k
    validation_losses = []  # List to store validation losses for each fold
    epoch_train_losses = []
    for fold in range(k):
        # Split the data into training and validation folds
        validation_data = train_data[fold * fold_size: (fold + 1) * fold_size]
        training_data = torch.cat([train_data[:fold * fold_size], train_data[(fold + 1) * fold_size:]])

        # Create data loaders for training and validation folds
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(validation_data, batch_size=batch_size)

        # Train the model for one epoch on the training data
        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            total_tested = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[:,1:].to(device), data[:,0].unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_tested += inputs.size(0)

            epoch_train_loss = running_loss / total_tested
            epoch_losses.append(epoch_train_loss)
            
            # Log training loss to wandb for each epoch without creating a chart
            wandb.log({"epoch": epoch, "k" + str(fold) + "_fold_training_loss": epoch_train_loss})
                
        # Log list of training losses to wandb for each fold
        epoch_train_losses.append(epoch_losses)
        wandb.log({'fold': fold, 'training_losses': epoch_losses})

        # Evaluate the model on the validation data
        with torch.no_grad():
            running_loss = 0.0
            total_tested = 0.0
            for data in valid_loader:
                inputs, labels = data[:,1:].to(device), data[:,0].unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total_tested += labels.size(0)

            # Log validation loss to wandb
            validation_loss = running_loss / total_tested
            wandb.log({'fold': fold, 'validation_loss': validation_loss})
            validation_losses.append(validation_loss)
            
    # Create summary chart that displays all the training losses for each fold
    wandb.log({"training_loss_chart" : wandb.plot.line_series(
        xs=list(range(epochs)),
        ys=epoch_train_losses,
        keys=["Fold " + str(fold) for fold in range(k)],
        title="Training Loss for each Fold",
        xname="Epoch"
    )})
    

    # Calculate the mean validation loss
    mean_validation_loss = sum(validation_losses) / len(validation_losses)
    wandb.log({'mean_validation_loss': mean_validation_loss}, commit=False)
    
    return model

if __name__ == "__main__":

    # Push to GPU if necessary
    gpu_number = "0"
    if gpu_number:
        import os
        use_gpu = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

    # Load the sweep configuration from YAML file
    with open('sweep.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, entity="naddeok", project="Rowan Rank Net")

    # Define training function
    def train(use_gpu):
        run = wandb.init()
        
        # Define hyperparameters
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        hidden_size = wandb.config.hidden_size
        epochs = wandb.config.epochs
        k = wandb.config.k
        
        # Load data
        _, train_data = process_data("US News - GB2023Engineering - Embargoed Until 3-29-22.xlsx")

        # Define model, loss function, and optimizer
        model = MLP(10, hidden_size, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate model
        model = k_fold_train(model, optimizer, criterion, k, train_data, batch_size, epochs, use_gpu)
        
        # Save 
        torch.save(model.state_dict(), "model_weights/" + run.name + '.pth')

    # Run sweep
    wandb.agent(sweep_id, lambda: train(use_gpu))
    

