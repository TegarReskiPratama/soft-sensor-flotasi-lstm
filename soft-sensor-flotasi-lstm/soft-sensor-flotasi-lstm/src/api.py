"""API inferensi untuk jendela sensor per jam; belum untuk operasi produksi."""
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import SENSOR


app = FastAPI(title="Soft Sensor Flotasi", version="1.0.0", description="Estimasi silika dataset bijih besi; demonstrasi metodologi.")


class Observasi(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jam: datetime
    sensor: dict[str, float]


class Permintaan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observasi: list[Observasi] = Field(min_length=6, max_length=6)


@lru_cache(maxsize=1)
def muat_model():
    import tensorflow as tf

    folder = Path(os.getenv("MODEL_DIR", "artifacts"))
    persiapan = joblib.load(folder / "preprocessor.joblib")
    model = tf.keras.models.load_model(folder / "model.keras", compile=False)
    if persiapan["fitur"] != SENSOR or persiapan["jam_input"] != 6:
        raise ValueError("Versi fitur atau panjang jendela pada model tidak cocok")
    return persiapan, model


@app.get("/health")
def health():
    folder = Path(os.getenv("MODEL_DIR", "artifacts"))
    ada = (folder / "preprocessor.joblib").is_file() and (folder / "model.keras").is_file()
    return {"status": "siap" if ada else "model_belum_tersedia"}


@app.post("/prediksi")
def prediksi(body: Permintaan):
    waktu = [o.jam for o in body.observasi]
    if any(t.tzinfo is not None for t in waktu):
        raise HTTPException(status_code=422, detail="Gunakan timestamp lokal tanpa zona waktu sesuai CSV sumber")
    if any(t.minute or t.second or t.microsecond for t in waktu):
        raise HTTPException(status_code=422, detail="Jam observasi harus tepat pada awal jam")
    if any(waktu[i+1] - waktu[i] != timedelta(hours=1) for i in range(5)):
        raise HTTPException(status_code=422, detail="Enam jam harus berurutan tanpa celah")
    if any(set(o.sensor) != set(SENSOR) for o in body.observasi):
        raise HTTPException(status_code=422, detail="Nama sensor harus tepat sesuai 19 fitur model")
    nilai = np.array([[o.sensor[n] for n in SENSOR] for o in body.observasi], dtype=float)
    if not np.isfinite(nilai).all():
        raise HTTPException(status_code=422, detail="Semua sensor harus berupa angka finite")
    try:
        prep, model = muat_model()
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {e}") from e
    if ((nilai < prep["batas_bawah"]) | (nilai > prep["batas_atas"])).any():
        raise HTTPException(status_code=422, detail="Sensor di luar rentang persentil 0,5–99,5 data train; prediksi ditahan")
    x = prep["skala"].transform(prep["imputer"].transform(nilai)).astype(np.float32)
    y = float(model.predict(x[np.newaxis, :, :], verbose=0)[0, 0])
    if not np.isfinite(y) or not 0 <= y <= 100:
        raise HTTPException(status_code=503, detail="Keluaran model di luar batas fisik; prediksi ditahan")
    return {
        "jam_target": (waktu[-1] + timedelta(hours=1)).isoformat(),
        "estimasi_persen_silika": y,
        "catatan": "Demonstrasi pada dataset bijih besi; perlu validasi domain sebelum penerapan.",
    }
