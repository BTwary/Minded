"""Vendor certification datasets loader with offline-first snapshot and SHA-256 integrity."""
import hashlib
import os
from typing import Dict, Optional
import pandas as pd

CERTIFICATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "certification")
)
CHECKSUMS_FILE = os.path.join(CERTIFICATION_DIR, "checksums.sha256")

EXPECTED_CHECKSUMS: Dict[str, str] = {
    "diamonds.csv": "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4",
    "flights.csv": "237d834127d9c6355630d8f443a7a2377b5925923010009b59809ba0b67f4fac",
    "mpg.csv": "c14b8b855ea7ee86cb9736bf8caaf281c4685ca08826f3eb2acaccaaf40f0d5a",
    "penguins.csv": "e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1",
    "planets.csv": "a6d10044887e17396974525a366f5fa2e4b34df70f491e64eb9943de0e3d3825",
    "taxis.csv": "08d6d71784dbaa2651fee37fc03389754194c05d72d2d19cbc2c799dea6ac09d",
    "tips.csv": "e54cc4d2ce1bff65d32ca60b3e4b802e06bde1d7e7caf6f796f6bf7370e863b0",
    "titanic.csv": "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2",
}


def verify_certification_data() -> Dict[str, bool]:
    """Verify SHA-256 checksums of vendored certification data files."""
    results = {}
    for filename, expected_hash in EXPECTED_CHECKSUMS.items():
        filepath = os.path.join(CERTIFICATION_DIR, filename)
        if not os.path.exists(filepath):
            results[filename] = False
            continue
        with open(filepath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        results[filename] = (actual_hash == expected_hash)
    return results


def load_dataset(name: str) -> pd.DataFrame:
    """Load a certification dataset offline-first from data/certification,
    falling back to online seaborn if the local cache is absent."""
    try:
        import seaborn as sns
        if os.path.isdir(CERTIFICATION_DIR):
            return sns.load_dataset(name, data_home=CERTIFICATION_DIR)
        return sns.load_dataset(name)
    except Exception:
        # Fallback to direct pandas parsing if seaborn unavailable
        csv_path = os.path.join(CERTIFICATION_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Post-process known types
            if name == "titanic":
                if "class" in df.columns:
                    df["class"] = pd.Categorical(df["class"], ["First", "Second", "Third"])
                if "deck" in df.columns:
                    df["deck"] = pd.Categorical(df["deck"], list("ABCDEFG"))
            elif name == "tips":
                df["day"] = pd.Categorical(df["day"], ["Thur", "Fri", "Sat", "Sun"])
                df["sex"] = pd.Categorical(df["sex"], ["Male", "Female"])
                df["time"] = pd.Categorical(df["time"], ["Lunch", "Dinner"])
                df["smoker"] = pd.Categorical(df["smoker"], ["Yes", "No"])
            elif name == "penguins":
                df["sex"] = df["sex"].str.title()
            elif name == "diamonds":
                df["color"] = pd.Categorical(df["color"], ["D", "E", "F", "G", "H", "I", "J"])
                df["clarity"] = pd.Categorical(df["clarity"], ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
                df["cut"] = pd.Categorical(df["cut"], ["Ideal", "Premium", "Very Good", "Good", "Fair"])
            elif name == "taxis":
                df["pickup"] = pd.to_datetime(df["pickup"])
                df["dropoff"] = pd.to_datetime(df["dropoff"])
            elif name == "flights":
                months = df["month"].str[:3]
                df["month"] = pd.Categorical(months, months.unique())
            return df
        raise


def load_all_certification_datasets(
    *,
    sample_diamonds: bool = True,
    rename_titanic_class: bool = True,
    diamonds_sample_n: int = 8000,
    diamonds_random_state: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Load all 8 standard certification datasets with consistent pre-processing."""
    names = ["titanic", "tips", "penguins", "mpg", "diamonds", "planets", "taxis", "flights"]
    frames = {}
    for n in names:
        df = load_dataset(n)
        if n == "diamonds" and sample_diamonds:
            df = df.sample(diamonds_sample_n, random_state=diamonds_random_state).reset_index(drop=True)
        if n == "titanic" and rename_titanic_class and "class" in df.columns:
            df = df.rename(columns={"class": "passenger_class"})
        frames[n] = df
    return frames
