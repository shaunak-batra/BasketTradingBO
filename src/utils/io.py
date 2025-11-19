"""Functions for saving and loading data in various formats."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_parquet(df: pd.DataFrame, file_path: str, **kwargs: Any) -> None:
    """Save DataFrame to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    **kwargs : Any
        Additional arguments passed to to_parquet()
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, **kwargs)


def load_parquet(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """Load DataFrame from Parquet format.

    Parameters
    ----------
    file_path : str
        Input file path
    **kwargs : Any
        Additional arguments passed to read_parquet()

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    return pd.read_parquet(file_path, **kwargs)


def save_hdf5(df: pd.DataFrame, file_path: str, key: str = "data", **kwargs: Any) -> None:
    """Save DataFrame to HDF5 format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    key : str
        HDF5 key (default: "data")
    **kwargs : Any
        Additional arguments passed to to_hdf()
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(file_path, key=key, mode='w', **kwargs)


def load_hdf5(file_path: str, key: str = "data", **kwargs: Any) -> pd.DataFrame:
    """Load DataFrame from HDF5 format.

    Parameters
    ----------
    file_path : str
        Input file path
    key : str
        HDF5 key (default: "data")
    **kwargs : Any
        Additional arguments passed to read_hdf()

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    return pd.read_hdf(file_path, key=key, **kwargs)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2, **kwargs: Any) -> None:
    """Save dictionary to JSON format.

    Parameters
    ----------
    data : Dict
        Dictionary to save
    file_path : str
        Output file path
    indent : int
        JSON indentation (default: 2)
    **kwargs : Any
        Additional arguments passed to json.dump()
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_json(file_path: str, **kwargs: Any) -> Dict[str, Any]:
    """Load dictionary from JSON format.

    Parameters
    ----------
    file_path : str
        Input file path
    **kwargs : Any
        Additional arguments passed to json.load()

    Returns
    -------
    Dict
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, **kwargs: Any) -> None:
    """Save DataFrame to CSV format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    **kwargs : Any
        Additional arguments passed to to_csv()
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, **kwargs)


def load_csv(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """Load DataFrame from CSV format.

    Parameters
    ----------
    file_path : str
        Input file path
    **kwargs : Any
        Additional arguments passed to read_csv()

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    return pd.read_csv(file_path, **kwargs)
