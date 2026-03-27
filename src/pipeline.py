import sys
from pathlib import Path

# Tested this code to ingest the data - Should work. Adding it here. - ECCLESIA

print("Current working directory:", Path.cwd())
print("Parent directory:", Path.cwd().parent)

# Get absolute path to dataset
sys.path.insert(0, str(Path.cwd().parent)) 
from src.ingest import load_kaggle_data