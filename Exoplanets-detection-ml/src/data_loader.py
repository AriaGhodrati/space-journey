# src/data_loader.py

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters
    ----------
    path : str
        Path to CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    df = pd.read_csv(path)

    return df
