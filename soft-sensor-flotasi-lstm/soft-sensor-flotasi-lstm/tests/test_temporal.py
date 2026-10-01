"""Uji kebocoran waktu dan konsistensi agregasi lintas chunk."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data import SENSOR, TARGET, baca_per_jam, bagi_waktu, jendela
from src.latih import transformasi


def contoh(n=100):
    jam = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({nama: np.arange(n, dtype=float) for nama in SENSOR}, index=jam)
    df[TARGET] = np.arange(n, dtype=float) / 10
    return df


def test_split_urutan_dan_preprocessing_train_saja():
    df = contoh()
    for nama in SENSOR:
        df.loc[df.index[70]:, nama] = 1_000_000.0
    train, val, test = bagi_waktu(df)
    assert train.index.max() < val.index.min() < test.index.min()
    (_, skala), fitur = transformasi((train, val, test))
    assert np.isclose(skala.mean_[0], np.arange(70).mean())
    assert fitur[1][0, 0] > 10
    x, y, _, t = jendela(train, fitur[0])
    assert x.shape[1:] == (6, 19)
    assert t.max() <= train.index.max()
    assert y[0] == pytest.approx(df.loc[df.index[6], TARGET])


def test_jendela_membuang_celah():
    df = contoh(25).drop(pd.Timestamp("2025-01-01 12:00:00"))
    x, _, _, t = jendela(df, df[SENSOR].to_numpy())
    assert len(x) == 12  # target 06..11 (6), 19..24 (6)
    assert pd.Timestamp("2025-01-01 13:00:00") not in t


def test_agregasi_chunk_setara_dan_desimal_koma(tmp_path):
    isi = [
        {"date": "2025-01-01 00:00:00", **{n: "1,0" for n in SENSOR}, TARGET: "2,0"},
        {"date": "2025-01-01 00:20:00", **{n: "3,0" for n in SENSOR}, TARGET: "4,0"},
        {"date": "2025-01-01 00:40:00", **{n: "5,0" for n in SENSOR}, TARGET: "6,0"},
    ]
    path = tmp_path / "mini.csv"
    pd.DataFrame(isi).to_csv(path, index=False)
    kecil, besar = baca_per_jam(path, chunksize=1), baca_per_jam(path, chunksize=100)
    pd.testing.assert_frame_equal(kecil, besar)
    assert kecil.iloc[0][TARGET] == pytest.approx(4.0)
    assert kecil.iloc[0][SENSOR[0]] == pytest.approx(3.0)
