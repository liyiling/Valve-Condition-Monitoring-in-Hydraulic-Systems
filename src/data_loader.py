from pathlib import Path
import pandas as pd
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DATA_DIR = PROJECT_ROOT / "data" / "condition+monitoring+of+hydraulic+systems"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensors.json"


def read_file_to_df(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        BASE_DATA_DIR / filename,
        sep="\t",
        header=None
    )

def load_sensor_data(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)  # str oder Path

    with open(config_path, "r", encoding="utf-8") as f:
        sensor_files = json.load(f)

    return {name: read_file_to_df(fname) for name, fname in sensor_files.items()}

'''
Es gibt 2205 Messungen. 
Jede Messung besteht aus Zeitreihen aller Sensoren und genau einer Zielgröße
(Valve Condition aus profile.txt).
'''

def load_target():
    profile = read_file_to_df("profile.txt")
    return profile.iloc[:, 1].reset_index(drop=True)

