![US News logo](https://www.ivycoach.com/content/uploads/2020/09/2021-US-News-College-Rankings.jpg)

# RowanRankNet

Welcome to RowanRankNet, the public GitHub repository for Rowan University. This repository contains the code and resources for training a neural network to predict the university's US News world ranking. The neural network uses a 3-layer architecture (input layer of size 10, hidden layer of size 10, and output layer of size 1), chosen based on the limited data and complexity of the prediction problem.

# Setting Up a Virtual Environment and Running MLP.py

This guide will walk you through the steps to create a new virtual environment, install the required dependencies specified in the `requirements.txt` file, and run the `MLP.py` script. The training data and model weights will be stored using W&B.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.6 or later
- Git
- W&B account (sign up [here](https://wandb.ai/site))

## Creating a Virtual Environment

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/<username>/<repository_name>.git
   ```

2. Navigate to the project directory.
    ```bash
    cd <repository_name>
    ```

3. Create a new virtual environment.
    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment.
    ```bash
    source venv/bin/activate
    ```

## Installing Dependencies

1. Install the dependencies specified in the `requirements.txt` file.
    ```bash
    pip3 install -r requirements.txt
    ```

## Running a K-Fold Cross Validation Hyperparameter Sweep

1. Run the `kfoldcross_hypersweep.py` script.

    ```bash
    python3 kfoldcross_hypersweep.py
    ```
    
    The kfoldcross_hypersweep.py script contains the code for training an MLP regression model using PyTorch and Weights & Biases (W&B) hyperparameter sweeps. The script demonstrates how to implement a multilayer perceptron (MLP) for regression with one hidden layer, ReLU activation function, and trains the model using stochastic gradient descent with mean squared error loss. The hyperparameter sweep is configured using a YAML file and employs the Bayesian search method.

By logging the model's performance and hyperparameters to W&B, the code enables easy tracking and comparison of different hyperparameter configurations, and allows for seamless collaboration with teammates.

This YAML file is used to configure the hyperparameter sweep:

```yaml
method: bayes
metric:
  name: mean_validation_loss
  goal: minimize
parameters:
  batch_size:
    values: [1, 2, 4, 8, 16, 32, 64, 128, 200]
  learning_rate:
    distribution: uniform
    min: 0.000001
    max: 1.0
  epochs:
    distribution: int_uniform
    min: 1
    max: 100
  k:
    value: 200
  hidden_size:
    distribution: int_uniform
    min: 5
    max: 512
early_terminate:
  type: hyperband
  min_iter: 10
  max_iter: 100
  eta: 3

```

## Running Multiple Sweeps

The `start_sweep.sh` script is a Bash script that automates the process of starting multiple sweeps in parallel. It starts four sweeps, where each sweep runs the `kfold_hypersweep.py` script using a specified GPU. The script uses two GPUs, with each GPU running two sweeps simultaneously. The output of each sweep is saved to a separate file named `sweep<i>.out`, where `<i>` is the index of the sweep (0 to 3).

Here is a breakdown of the script:

1. The script iterates from 0 to 3 using a for loop.
2. It checks if the current loop index is even or odd. If it's even, it assigns GPU number 4 to the `gpu_number` variable. If it's odd, it assigns GPU number 5 to the `gpu_number` variable.
3. The `kfold_hypersweep.py` script is then run with the specified `gpu_number` using the `nohup` command, which allows the script to continue running even if the terminal is closed. The script is run in the background by appending `&` to the command.
4. The output of the `kfold_hypersweep.py` script is redirected to a file named `sweep<i>.out`, where `<i>` is replaced by the current loop index (0 to 3).
5. After all sweeps are started, the script waits for them to finish using the `wait` command before exiting.

To run multiple sweeps using the `start_sweep.sh` script, simply execute the script in your terminal:

```bash
./start_sweep.sh
```

Make sure the script has execute permissions. If not, you can add them with:

```bash
chmod +x start_sweep.sh
```

## Training Optimal Model

Once the optimal set of hyperparameters is found from the sweep, you can update the `config` dictionary with the new values. The trained model is then saved as a PyTorch state dictionary in a file named after the W&B run ID.

To add the optimal set of hyperparameters to the `config` dictionary in `train_model.py`, replace the existing values with the optimal values. For example, if the sweep found the optimal hyperparameters to be:

```python
config = {
    "batch_size": 100,
    "learning_rate": 0.01,
    "hidden_size": 500,
    "epochs": 50,
}
```

After updating the `config` dictionary with the optimal hyperparameters, run the `train_model.py` script to train the model using the new hyperparameters:

```bash
python3 train_model.py
```

This will train the model using the updated hyperparameters and save the trained model as a PyTorch state dictionary in the `model_weights` directory with a filename matching the W&B run ID.

## Making Predictions

In `interactive.py`, the script creates a graphical user interface (GUI) to interact with the trained model. To use the saved model in `model_weights` and the updated `hidden_size`, you need to modify the script with the correct `hidden_size` and `model_weights_path`.

```python
model = MLP(input_size=10, hidden_size=464, output_size=1)
```

Also, update the `model_weights_path` variable to point to the correct saved model file in the `model_weights` directory:

```python
model_weights_path = "model_weights/your_saved_model_filename.pth"
```

Replace `your_saved_model_filename.pth` with the actual file name of the saved model. Once the script is updated, you can run the script to launch the GUI.

If you're using Windows Subsystem for Linux (WSL), activate XLaunch before running the script. This will allow you to run the GUI application through WSL. To launch the GUI script, simply run:

```bash
python3 interactive.py
```
Below is an exaple of the GUI.

![GUI](example_of_gui.png)

These default values correspond to Rowan University's metrics. 

## Cross-validation with k-fold

We use k-fold cross-validation to ensure robust evaluation of the model's performance, splitting the dataset into training and validation sets. The model is trained and evaluated k times, with each subset serving as the validation set once and the remaining subsets as the training set. The final model performance is computed by averaging the metrics across the k iterations.

## Publicly Available Data and Relevant Sources for Research Papers

In addition to the data and resources provided in this repository, there are several publicly available data sources and relevant sources for research papers on university rankings. These include:

- **[Times Higher Education (THE) World University Rankings](https://www.timeshighereducation.com/world-university-rankings/methodology-world-university-rankings-2021):** A widely recognized ranking system that evaluates universities based on several criteria, including research, teaching, and international outlook.

- **[QS World University Rankings](https://www.topuniversities.com/qs-world-university-rankings/methodology):** A ranking system that evaluates universities based on several criteria, including academic reputation, employer reputation, and citations per faculty.

- **[Center for World University Rankings (CWUR)](https://cwur.org/methodology/world-university-rankings.php):** A ranking system that evaluates universities based on several criteria, including research output, quality of education, and alumni employment.

- **[Integrated Postsecondary Education Data System (IPEDS)](https://nces.ed.gov/ipeds/use-the-data):** A comprehensive data source for information on colleges and universities in the United States.

We encourage researchers and practitioners to explore these resources and incorporate them into their research and analysis. 

## Usage

The repository includes code for data preprocessing, model training and evaluation, hyperparameter tuning, and k-fold cross-validation. We provide documentation on ethical guidelines and regulations for conducting research, as well as permissions and agreements for using the ranking data.

We welcome contributions and collaborations from the Rowan community and external researchers and practitioners. Feel free to explore the code and resources and contact the repository owners with any questions or feedback.
