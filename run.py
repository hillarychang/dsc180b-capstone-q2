import sys
import json
from src.etl import get_data
from src.features import build_features
from src.model_training import train_model

def main(targets):
    pass

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)