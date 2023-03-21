# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from data_processing import process_data
from MLP import MLP

def standard_train(hidden_size, criterion, learning_rate, train_data, batch_size, epochs, use_gpu):
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Initalize model
    model = MLP(10, hidden_size, 1)
    model.to(device)
    
    # Initalize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders for training and validation folds
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
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
        wandb.log({"final_epoch": epoch, "final_training_loss": epoch_train_loss})    
        
    return model

if __name__ == "__main__":

    # Push to GPU if necessary
    gpu_number = "4"
    if gpu_number:
        import os
        use_gpu = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    
    # Define hyperparameters
    config = {  "batch_size"    : 200,
                "learning_rate" : 0.9988,
                "hidden_size"   : 296,
                "epochs"        : 7
                }

    # Initialize W&B
    run = wandb.init(config=config, entity="naddeok", project="Rowan Rank Net")
                
    # Load data
    _, train_data = process_data("US News - GB2023Engineering - Embargoed Until 3-29-22.xlsx")

    # Define model, loss function, and optimizer
    criterion = nn.MSELoss()
    
    # Train on all
    model = standard_train(wandb.config.hidden_size, criterion, wandb.config.learning_rate, train_data, wandb.config.batch_size, wandb.config.epochs, use_gpu)
    
    # Save 
    torch.save(model.state_dict(), "model_weights/" + run.name + '.pth')

    

