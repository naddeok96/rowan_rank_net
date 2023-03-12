import torch
import pandas as pd

def process_data(filename):
    # Read the Excel file and set the fourth row as header
    df = pd.read_excel(filename, header=4)

    # Delete the first two columns
    df = df.iloc[:, 2:]

    # Replace NaNs in the first column with zeros
    df.iloc[:, 0] = df.iloc[:, 0].fillna(0)

    # Convert NaNs and "N\A"s to None
    df = df.replace({"NaN": None, "N\A": None})

    # Convert all integers and strings to floats
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Remove all occurrences of "$" and "%" from string elements
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.replace('$', '').replace('%', '') if isinstance(x, str) else x)
    
    # Convert the Pandas DataFrame to a PyTorch tensor
    return torch.tensor(df.values, dtype=torch.float32)
