import pandas as pd
import argparse 

def drop_columns(df: pd.DataFrame, columns_to_drop: list = []) -> pd.DataFrame:
    """
    Drop specified columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which columns need to be dropped.
    columns_to_drop (list): A list of column names to be dropped.

    Returns:
    pd.DataFrame: A new DataFrame with specified columns dropped.
    """
    columns = ["segmentation.class_name", 'segmentation.external_id', 'segmentation.$stable', "segmentation.visit", "segmentation.stable", "segmentation.platform"]
    existed_columns = [col for col in columns if col in df.columns]
    columns_to_drop = [col for col in existed_columns if col in df.columns]
    return df.drop(columns=columns_to_drop, errors='ignore')

def rename_columns(
    df: pd.DataFrame, 
    rename_dict: dict = {
        "_id.$oid": "record_id",
        "created_at.$date": "created_at",
        "dur": "duration"
    }) -> pd.DataFrame:
    """
    Rename columns in a DataFrame based on a provided dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns need to be renamed.
    rename_dict (dict): A dictionary where keys are current column names and values are new column names.

    Returns:
    pd.DataFrame: A new DataFrame with renamed columns.
    """
    return df.rename(columns=rename_dict)


def map_values(df: pd.DataFrame, column_names: list[str] = ["key"], mapping_dict: dict[str, dict] = {
    "key": {
    "[CLY]_view": "View",
    "action": "Action"
    }
}) -> pd.DataFrame:
    """
    Map values in a specified column of a DataFrame based on a provided dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be mapped.
    column_name (str): The name of the column whose values need to be mapped.
    mapping_dict (dict): A dictionary where keys are current values and values are new values.

    Returns:
    pd.DataFrame: A new DataFrame with mapped values in the specified column.
    """
    for column_name in column_names:
        if column_name in df.columns:
            df[column_name] = df[column_name].map(mapping_dict[column_name], na_action='ignore')
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean and preprocess data.")
    parser.add_argument("--input", help="Path to the input CSV file.")
    parser.add_argument("--output", help="Path to the output CSV file.")
    args = parser.parse_args()

    # Read the input CSV
    df = pd.read_csv(args.input)

    # Drop unwanted columns
    df = drop_columns(df, [])

    # Rename columns
    df = rename_columns(df)

    # Map values in specified columns
    df = map_values(df)

    # Save the cleaned DataFrame to a new CSV
    df.to_csv(args.output, index=False)