![US News logo](https://www.ivycoach.com/content/uploads/2020/09/2021-US-News-College-Rankings.jpg)

# RowanRankNet

Welcome to RowanRankNet, the private GitHub repository for Rowan University. This repository contains the code and resources for training a neural network to predict the university's US News world ranking. The neural network uses a 3-layer architecture (input layer of size 10, hidden layer of size 10, and output layer of size 1), chosen based on the limited data and complexity of the prediction problem.

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

## Running the Script

1. Run the MLP.py script.
    ```bash
    python3 MLP.py
    ```
    The training data and model weights will be stored using W&B.

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
