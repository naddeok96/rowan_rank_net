# RowanRankNet

Welcome to RowanRankNet, the private GitHub repository for Rowan University. This repository contains the code and resources for training a neural network to predict the university's US News world ranking. The neural network uses a 3-layer architecture (input layer of size 10, hidden layer of size 10, and output layer of size 1), chosen based on the limited data and complexity of the prediction problem.

![US News logo](https://www.ivycoach.com/content/uploads/2020/09/2021-US-News-College-Rankings.jpg)

## Cross-validation with k-fold

We use k-fold cross-validation to ensure robust evaluation of the model's performance, splitting the dataset into training and validation sets. The model is trained and evaluated k times, with each subset serving as the validation set once and the remaining subsets as the training set. The final model performance is computed by averaging the metrics across the k iterations.

## Usage

The repository includes code for data preprocessing, model training and evaluation, hyperparameter tuning, and k-fold cross-validation.

We welcome contributions and collaborations from the Rowan community and external researchers and practitioners. Feel free to explore the code and resources and contact the repository owners with any questions or feedback.
