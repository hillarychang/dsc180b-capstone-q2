import re
import pandas as pd

def clean_text(data, column="memo"):
    """
    Cleans text data in a specified column of a DataFrame by applying multiple regex patterns:
    - Removes dates in 'MM-DD' or 'MM-DD-YYYY' format, optionally preceded by 'CA'.
    - Removes excessive 'X' characters (unless preceded by '#').
    - Strips unnecessary punctuation.
    - Removes state abbreviations at the end of entries.
    - Removes phrases like 'POS WITHDRAWAL', 'DEBIT CARD WITHDRAWAL', and 'PURCHASE'.
    - Converts all text to lowercase and removes excess whitespace.

    Parameters:
    ----------
    data : pd.DataFrame
        Input DataFrame containing the text data to be cleaned.
    column : str
        The name of the column in `data` to clean. Defaults to 'memo'.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the cleaned text data.
    """
    # Check if data is a DataFrame and contains the specified column
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input `data` must be a pandas DataFrame.")
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    # Make a copy of the data to avoid modifying the original DataFrame
    df = data.copy()

    # Make sure the data that we are doing doesn't have the same memos and categories
    df = df[df['memo'] != df['category']]

    # Define regex patterns
    pattern1 = r"\b(?:CA\s+)?(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])(?:/\d{2,4})?\b"  # Dates and optional 'CA'
    pattern2 = r"(?<!#)X+|#X+"  # Excessive 'X' characters
    pattern3 = r"[^a-zA-Z0-9\s./]"  # Unnecessary punctuation
    pattern4 = r"\s[A-Z]{2}$"  # State codes at the end
    pattern5 = r"(pos withdrawal|debit card withdrawal)"  # Transaction phrases
    pattern6 = r"(purchase)"  # 'purchase' phrase

    # Apply regex transformations to the specified column
    df[column] = df[column].apply(
        lambda x: re.sub(pattern3, "", re.sub(pattern2, "", re.sub(pattern1, "", x)))
    )
    df[column] = df[column].apply(lambda x: " ".join(x.split()).strip())
    df[column] = df[column].apply(lambda x: re.sub(pattern4, "", x))
    df[column] = df[column].apply(
        lambda x: re.sub(pattern6, "", re.sub(pattern5, "", x.lower()))
    )

    return df
