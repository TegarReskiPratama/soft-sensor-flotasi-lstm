"""Pembacaan hemat memori, agregasi per jam, dan pemisahan temporal."""
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR = [
    "Starch Flow", "Amina Flow", "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density",
    *[f"Flotation Column {i:02d} Air Flow" for i in range(1, 8)],
    *[f"Flotation Column {i:02d} Level" for i in range(1, 8)],
]
TARGET = "% Silica Concentrate"
NUMERIK = [*SENSOR, TARGET]


def baca_per_jam(csv: str | Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Rata-rata berbobot per jam; baris lintas chunk tetap terhitung benar."""
    if chunksize <= 0:
        raise ValueError("chunksize harus positif")
    total = None
    jumlah = None
    pembaca = pd.read_csv(csv, usecols=["date", *NUMERIK], chunksize=chunksize, low_memory=False, dtype=str)
    for chunk in pembaca:
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.loc[chunk["date"].notna()].copy()
        chunk["jam"] = chunk["date"].dt.floor("h")
        for kolom in NUMERIK:
            chunk[kolom] = pd.to_numeric(chunk[kolom].str.replace(",", ".", regex=False), errors="coerce")
        nilai = chunk.groupby("jam", sort=True)[NUMERIK]
        part_total = nilai.sum(min_count=1).fillna(0)
        part_jumlah = nilai.count()
        total = part_total if total is None else total.add(part_total, fill_value=0)
        jumlah = part_jumlah if jumlah is None else jumlah.add(part_jumlah, fill_value=0)
    if total is None:
        raise ValueError("CSV kosong")
    hasil = total.div(jumlah.replace(0, np.nan)).sort_index()
    hasil.index.name = "jam"
    hasil = hasil.loc[hasil[TARGET].notna()]
    if hasil.empty:
        raise ValueError("Tidak ada jam dengan target lab yang valid")
    return hasil


def bagi_waktu(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Proporsi split tidak sah")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Indeks jam harus unik dan terurut")
    a, b = int(len(df) * train), int(len(df) * (train + val))
    if min(a, b-a, len(df)-b) < 2:
        raise ValueError("Terlalu sedikit jam pada salah satu split")
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def jendela(df: pd.DataFrame, fitur: np.ndarray, panjang: int = 6):
    """Target di t+1, hanya saat semua cap waktu t-panjang+1...t+1 berurutan."""
    if panjang < 1 or fitur.shape != (len(df), len(SENSOR)):
        raise ValueError("Bentuk fitur atau panjang jendela tidak sah")
    x, y, lab_terakhir, waktu = [], [], [], []
    jam = df.index
    target = df[TARGET].to_numpy(dtype=np.float32)
    for akhir in range(panjang-1, len(df)-1):
        awal = akhir - panjang + 1
        rentang = jam[awal:akhir+2]
        nanodetik = rentang.to_numpy(dtype="datetime64[ns]").view(np.int64)
        if not (np.diff(nanodetik) == 3_600_000_000_000).all():
            continue
        if not (np.isfinite(target[akhir]) and np.isfinite(target[akhir+1])):
            continue
        x.append(fitur[awal:akhir+1])
        y.append(target[akhir+1])
        lab_terakhir.append(target[akhir])
        waktu.append(jam[akhir+1])
    if not x:
        raise ValueError("Tidak ada jendela valid: periksa rentang waktu dan data")
    return (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(lab_terakhir, dtype=np.float32), pd.DatetimeIndex(waktu))
