"""Unduh dataset publik tanpa memasukkan berkas mentah ke repositori."""
from pathlib import Path
import shutil


def main() -> None:
    import kagglehub

    sumber = Path(kagglehub.dataset_download("edumagalhaes/quality-prediction-in-a-mining-process"))
    kandidat = list(sumber.rglob("MiningProcess_Flotation_Plant_Database.csv"))
    if len(kandidat) != 1:
        raise FileNotFoundError(f"CSV yang diharapkan tidak ditemukan di {sumber}")
    tujuan = Path("data/MiningProcess_Flotation_Plant_Database.csv")
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kandidat[0], tujuan)
    print(f"Data tersedia di {tujuan} ({tujuan.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
