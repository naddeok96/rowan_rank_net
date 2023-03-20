import pandas as pd
import torch

def process_data(filename: str) -> tuple:
    """
    Read an Excel file, preprocess the data, and convert it to a PyTorch tensor.

    Args:
        filename (str): The name of the Excel file to read.

    Returns:
        A tuple containing two objects:
        1. A Pandas DataFrame with the preprocessed data.
        2. A PyTorch tensor with the preprocessed data.
    """
    # Read the Excel file and set the fourth row as header
    df = pd.read_excel(filename, header=4)

    # Delete the first two columns
    df = df.iloc[:, 2:]

    # Replace NaNs in the first column with zeros
    df.iloc[:, 0] = df.iloc[:, 0].fillna(0)

    # Convert NaNs and "N\A"s to None
    df = df.replace({"NaN": None, "N\A": None})
    
    # Replace non-numeric characters with empty strings
    df = df.replace('[^0-9]', '', regex=True)
    
    # Convert all integers and strings to floats
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))     

    # Replace None values with column averages
    df = df.fillna(df.mean())
    
    # Convert the Pandas DataFrame to a PyTorch tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)
    
    return df, tensor


if __name__ == "__main__":
    df, _ = process_data("US News - GB2023Engineering - Embargoed Until 3-29-22.xlsx")
    df.to_excel("TEMP.xlsx", index=False)