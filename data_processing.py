# Imports 
import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def process_data(filename: str) -> tuple:
    """
    Read an Excel file, preprocess the data, and save the preprocessed data as a PyTorch tensor.

    Args:
        filename (str): The name of the Excel file to read.
        save_dir (str): The directory to save the preprocessed data to.

    Returns:
        A tuple containing two objects:
        1. A Pandas DataFrame with the preprocessed data.
        2. The file path to the saved PyTorch tensor.
    """
    # Read the Excel file and set the fourth row as header
    df = pd.read_excel(filename, header=4)

    # Delete the first two columns
    df = df.iloc[:, 2:]

    # Replace NaNs in the first column with tens
    df.iloc[:, 0] = df.iloc[:, 0].fillna(10)

    # Replace NaNs in the "2021 faculty membership in National Academy of Engineering" column with zeros
    df.iloc[:, 6] = df.iloc[:, 6].fillna(0)

    # Replace
    df["Overall score"] = df["Overall score"] / 100

    # Convert NaNs and "N\A"s to None
    df = df.replace({"NaN": None, "N\A": None})
    
    # Replace  all non-numeric characters and non-decimal-point characters with an empty string
    df = df.replace('[^0-9.]', '', regex=True)

    # Convert all integers and strings to floats
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))     
    
    # Group by the first column
    groups = df.groupby(df.columns[0])

    # Replace "none" values
    for column in df.columns:
        if column == df.columns[0]:
            continue

        # Group by the unique values in the first column, then replace the None (NaN) values in the current column
        # with the mean of the respective group (in the current column)
        df[column] = df.groupby(df.columns[0])[column].transform(lambda x: x.fillna(x.mean()))

        # Find the indices of remaining None values in the current column
        none_indices = df[df[column].isnull()].index

        # For each None value, calculate the mean of the first three non-None values above and below
        # and replace the None value with the calculated mean
        for idx in none_indices:
            available_values = df.loc[max(0, idx - 3):min(df.shape[0] - 1, idx + 3), column].dropna()
            if not available_values.empty:
                mean_value = available_values.mean()
                df.loc[idx, column] = mean_value

    # Assert no None values are left in the DataFrame
    assert not df.isnull().values.any(), "There are still None values in the DataFrame"


    # apply min-max normalization to all columns except the first one
    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    # create a new dataframe to store the min and max values for each column
    min_max_df = pd.DataFrame({
        'column': df.columns[1:],
        'min_value': scaler.data_min_,
        'max_value': scaler.data_max_
    })

    # save the min and max dataframe to an Excel file
    min_max_df.to_excel('data/min_max_values.xlsx', index=False)
    df.to_excel('data/preprocessed_data.xlsx', index=False)
    
    # Convert the Pandas DataFrame to a PyTorch tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)

    # Save the tensor to a file
    tensor_file_path = os.path.join("data/preprocessed_data.pt")
    torch.save(tensor, tensor_file_path)
    
    return df, tensor

if __name__ == "__main__":
    process_data("data/US News - GB2023Engineering - Embargoed Until 3-29-22.xlsx")