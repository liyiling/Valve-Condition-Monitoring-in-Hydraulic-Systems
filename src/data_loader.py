from pathlib import Path
import pandas as pd
import json


BASE_DATA_DIR = Path("data/condition+monitoring+of+hydraulic+systems")

def read_file_to_df(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        BASE_DATA_DIR / filename,
        sep="\t",
        header=None
    )

def load_sensor_data(config_path="config/sensors.json"):
    with open(config_path, "r") as f:
        sensor_files = json.load(f)

    return {
        name: read_file_to_df(fname)
        for name, fname in sensor_files.items()
    }

'''
Es gibt 2205 Messungen. 
Jede Messung besteht aus Zeitreihen aller Sensoren und genau einer Zielgröße
(Valve Condition aus profile.txt).
'''

def load_target():
    profile = read_file_to_df("profile.txt")
    return profile.iloc[:, 1].reset_index(drop=True)

