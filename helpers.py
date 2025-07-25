import os
import numpy as np
import pandas as pd
from typing import Optional


def remove_row_if_nan(df: pd.DataFrame, col: str, verbose: bool = False) -> None:
    """
    Removes rows from df in-place where df[col] is NaN, inf, or -inf.
    Prints the number of rows dropped.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify in-place.
        col (str): The column name to check.
        verbose (bool): If True, prints the number of rows dropped (default: False).

    Returns:
        None
    """
    initial_rows = len(df)

    # Remove NaN rows
    df.dropna(subset=[col], inplace=True)
    after_nan_removal = len(df)
    nan_dropped = initial_rows - after_nan_removal

    # Remove inf and -inf rows
    df.drop(df[df[col].isin([float('inf'), float('-inf')])].index, inplace=True)
    final_rows = len(df)
    inf_dropped = after_nan_removal - final_rows

    total_dropped = initial_rows - final_rows

    if verbose and total_dropped > 0:
        print(f"Dropped {total_dropped:,} rows from column '{col}' ({nan_dropped:,} NaN, {inf_dropped:,} inf/-inf)")


def remove_rows_by_col_value(
    df: pd.DataFrame,
    col: str,
    min_value: float,
    drop_null: bool = False,
    verbose: bool = False
) -> None:
    """
    Removes rows from df in-place where df[col] < min_value.
    If drop_null is True, also drops rows where df[col] is NaN or infinite.
    Prints the number of rows dropped.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify in-place.
        col (str): The column name to check.
        min_value (float): The minimum value threshold.
        drop_null (bool): If True, also drop rows where the column value is NaN or infinite (default: false).
        verbose (bool): If True, prints the number of rows dropped (default: False).

    Returns:
        None
    """
    initial_rows = len(df)

    # Drop null/infinite values if requested
    null_dropped = 0
    if drop_null:
        before_null = len(df)
        remove_row_if_nan(df, col)
        null_dropped = before_null - len(df)

    # Drop rows below minimum value
    before_min_filter = len(df)
    df.drop(df[df[col] < min_value].index, inplace=True)
    min_value_dropped = before_min_filter - len(df)

    total_dropped = initial_rows - len(df)

    if verbose and total_dropped > 0:
        if drop_null and null_dropped > 0:
            print(f"Dropped {total_dropped:,} rows from column '{col}' ({null_dropped:,} null/inf, {min_value_dropped:,} < {min_value})")
        else:
            print(f"Dropped {min_value_dropped:,} rows from column '{col}' (< {min_value})")


def add_floored_column(
    df: pd.DataFrame,
    col: str,
    floor_value: float
) -> None:
    """
    Adds a new column to the DataFrame named "{col}_FLOORED".
    The new column's value is the same as the original column if it is greater than floor_value,
    otherwise it is set to floor_value.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify in-place.
        col (str): The column to apply the floor to.
        floor_value (float): The minimum value allowed in the new column.

    Returns:
        None
    """
    new_col = f"{col}_FLOORED"
    df[new_col] = df[col].apply(lambda x: x if x > floor_value else floor_value)


def add_ratio_column(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> None:
    """
    Adds a new column to the DataFrame named "D_ME_{col1}_{col2}" (removing any 'IQ_' or 'D_ME_' prefix and '_FISCAL' suffix from col1/col2).
    The new column's value is df[col1] / df[col2] if df[col2] != 0, else df[col1].

    Parameters:
        df (pd.DataFrame): The DataFrame to modify in-place.
        col1 (str): The numerator column name.
        col2 (str): The denominator column name.

    Returns:
        None
    """
    def clean_col_name(col):
        for prefix in ["IQ_", "D_ME_", "TOTAL_"]:
            if col.startswith(prefix):
                col = col.removeprefix(prefix)
        for suffix in ["_FISCAL", "_FLOORED"]:
            if col.endswith(suffix):
                col = col.removesuffix(suffix)
        return col

    col1_clean = clean_col_name(col1)
    col2_clean = clean_col_name(col2)

    new_col = f"D_ME_{col1_clean}_{col2_clean}"
    df[new_col] = df.apply(
        lambda row: row[col1] / row[col2] if row[col2] != 0 else row[col1],
        axis=1
    )


def add_growth_column(
    df: pd.DataFrame,
    col: str
) -> None:
    """
    Adds a new column to the DataFrame named "D_ME_GR_{col}".
    Logic:
        - If both col and col_LAG1Y are 0, set to 0.
        - If col_LAG1Y is 0 (but col is not), set to mean(col) + 30 * std(col).
        - Else, set to col / col_LAG1Y - 1.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify in-place.
        col (str): The current period column name.

    Returns:
        None
    """
    lag_col = f"{col}_LAG1Y"
    std_val = df[col].std(skipna=True)
    mean_val = df[col].mean(skipna=True)
    col_tmp = col.removeprefix('IQ_')
    new_col = f"D_ME_GR_{col_tmp.removeprefix('TOTAL_')}"


    df[new_col] = df.apply(
        lambda row: 0 if (row[col] == 0 and row[lag_col] == 0)
        else (mean_val + 30 * std_val if row[lag_col] == 0 else row[col] / row[lag_col] - 1),
        axis=1
    )


def prep_greg_data(start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   force_reload: bool = False
) -> pd.DataFrame:
    """
    Loads and prepares the Greg data for analysis.

    Steps performed:
        - Reads the 'datapull_20250725.csv' file into a DataFrame.
        - Converts the 'DAY_DATE' column to datetime and creates a 'YEAR_MONTH' period column.
        - Optionally filters data between start_date and end_date if provided.
        - Removes rows where 'IQ_EBITDA' null/infinite, and where 'MARKET_CAP_FISCAL' is less than 100.
        - Adds floored columns for 'IQ_TOTAL_DEBT' and 'IQ_TOTAL_ASSETS' (minimum value of 1).
        - Computes free cash flow ('D_ME_FCF') and log of floored assets ('D_ME_LOG_ASSETS').
        - Adds several ratio columns for financial metrics.
        - Adds growth columns for revenue, EBITDA, net income, and total assets.
        - Creates INDUSTRY_SECTOR_W_BIOTECH column with biotechnology classification.

    Parameters:
        start_date (Optional[str]): Start date in 'YYYY-MM' format (inclusive). If None, no start filter applied.
        end_date (Optional[str]): End date in 'YYYY-MM' format (inclusive). If None, no end filter applied.
        force_reload (bool): If True, forces reprocessing even if cached file exists (default: False).

    Returns:
        pd.DataFrame: The cleaned and feature-engineered DataFrame.
    """
    # Create cache filename based on date filters
    cache_suffix = ""
    if start_date is not None or end_date is not None:
        start_str = start_date if start_date is not None else "all"
        end_str = end_date if end_date is not None else "all"
        cache_suffix = f"_{start_str}_to_{end_str}"

    cache_file = f"data/prep_greg_data{cache_suffix}.parquet"

    # Check if cached file exists and we're not forcing reload
    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached data from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Processing data and saving to {cache_file}")

    df = pd.read_csv("data/datapull_20250725.csv")

    # Create INDUSTRY_SECTOR_W_BIOTECH column
    df["INDUSTRY_SECTOR_W_BIOTECH"] = df["INDUSTRY_SECTOR"].copy()
    if "SECTOR_BIOTECH_PHARMA" in df.columns:
        df.loc[df["SECTOR_BIOTECH_PHARMA"] == 1, "INDUSTRY_SECTOR_W_BIOTECH"] = "Biotechnology"

    df["DAY_DATE"] = pd.to_datetime(df["DAY_DATE"])
    df["YEAR_MONTH"] = df["DAY_DATE"].dt.to_period("M")

    # Apply date filtering if parameters are provided
    if start_date is not None or end_date is not None:
        if start_date is not None:
            df = df[df["YEAR_MONTH"] >= start_date]
        if end_date is not None:
            df = df[df["YEAR_MONTH"] <= end_date]

    print(f"Number of rows before filtering: {len(df)}")

    remove_row_if_nan(df, "IQ_EBITDA")
    remove_rows_by_col_value(df, "MARKET_CAP_FISCAL", 100)

    print(f"Number of rows after filtering: {len(df)}")

    add_floored_column(df, "IQ_TOTAL_DEBT", 1)
    add_floored_column(df, "IQ_TOTAL_ASSETS", 1)

    df["D_ME_FCF"] = df["IQ_CASH_OPER"] - df["IQ_CAPEX"]
    df["D_ME_LOG_ASSETS"] = np.log(df["IQ_TOTAL_ASSETS_FLOORED"])

    add_ratio_column(df, "IQ_EBITDA", "TEV_FISCAL")
    add_ratio_column(df, "D_ME_FCF", "IQ_EBITDA")
    add_ratio_column(df, "IQ_EBITDA", "IQ_GP")
    add_ratio_column(df, "IQ_GP", "IQ_TOTAL_REV")
    add_ratio_column(df, "IQ_TOTAL_DEBT", "MARKET_CAP_FISCAL")
    add_ratio_column(df, "IQ_GP", "IQ_TOTAL_ASSETS_FLOORED")

    add_growth_column(df, "IQ_TOTAL_REV")
    add_growth_column(df, "IQ_EBITDA")
    add_growth_column(df, "IQ_NI")
    add_growth_column(df, "IQ_TOTAL_ASSETS")

    # Save processed data to cache
    os.makedirs("data", exist_ok=True)
    df.to_parquet(cache_file, index=False)
    print(f"Processed data saved to {cache_file}")

    return df
