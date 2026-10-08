"""Download the Online Retail II dataset from the UCI repository into data/raw/."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW_DIR)
    print("Done:", [p.name for p in RAW_DIR.iterdir()])
