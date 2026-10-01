"""Cetak payload ilustrasi yang memenuhi skema API, tanpa data tambang asli."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib

from .data import SENSOR


def main():
    awal = datetime(2025, 1, 1, 0)
    path = Path("artifacts/preprocessor.joblib")
    if not path.is_file():
        raise SystemExit("Latih model lebih dulu agar contoh memakai median sensor data train")
    persiapan = joblib.load(path)
    median = dict(zip(SENSOR, map(float, persiapan["imputer"].statistics_)))
    print(json.dumps({"observasi": [
        {"jam": (awal + timedelta(hours=i)).isoformat(), "sensor": median}
        for i in range(6)
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
