import sys
import json
import argparse
import pandas as pd
from src.etl import get_data
from src.features import build_features
from src.model_training import train_model
from src.text_cleaner import clean_text

def clean_data(data):
    """
    Cleans the 'memo' column of the data
    """
    cleaned_df = clean_text(data)
    return cleaned_df

def main(targets):
    pass

if __name__ == '__main__':
    # Create Argument Parser in order to read in targets
    parser = argparse.ArgumentParser(description = "Used to Run Model")
    parser.add_argument('dataset', choices = ['inflows', 'outflows'], help = "Specify the Dataset that should be Cleaned")
    args = parser.parse_args()

    # Determine which Dataset should be cleaned
    if args.dataset == 'inflows':
        file_name = "../data/ucsd-inflows.pqt"
    else:
        file_name = "../data/ucsd-outflows.pqt"
    
    # Read in and clean the data
    data = pd.read_parquet(file_name)
    cleaned_data = clean_data(data)
