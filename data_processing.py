import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

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

    # Replace NaNs in the first column with tens
    df.iloc[:, 0] = df.iloc[:, 0].fillna(10)

    # Replace
    df["Overall score"] = df["Overall score"] / 100

    # Convert NaNs and "N\A"s to None
    df = df.replace({"NaN": None, "N\A": None})
    
    # Replace  all non-numeric characters and non-decimal-point characters with an empty string
    df = df.replace('[^0-9.]', '', regex=True)

    # Convert all integers and strings to floats
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))     
    
    # Replace None values with column averages
    df = df.fillna(df.mean())

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
    min_max_df.to_excel('min_max_values.xlsx', index=False)
    
    # Convert the Pandas DataFrame to a PyTorch tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)
    
    return df, tensor

if __name__ == "__main__":
    df, _ = process_data("US News - GB2023Engineering - Embargoed Until 3-29-22.xlsx")
    df.to_excel("TEMP.xlsx", index=False)