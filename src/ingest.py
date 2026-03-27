import pandas as pd
from pathlib import Path


def load_kaggle_data():
    
    data_path = Path(__file__).resolve().parents[1] /"data" / "raw" / "dataset.csv"
    print(data_path)

    if not data_path.exists():
       raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df
    

def load_Spotify_data():
    pass