"""Eksperimen kronologis: baseline, LSTM, evaluasi, dan artefak."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .data import SENSOR, TARGET, baca_per_jam, bagi_waktu, jendela


def skor(y: np.ndarray, pred: np.ndarray) -> dict:
    galat = np.abs(np.asarray(y) - np.asarray(pred))
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "p90_galat_absolut": float(np.percentile(galat, 90)),
    }


def transformasi(splits):
    """Fit imputasi/skala hanya pada baris train; target tak pernah masuk input."""
    train = splits[0][SENSOR].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if np.isnan(train).all(axis=0).any():
        kosong = [SENSOR[i] for i in np.where(np.isnan(train).all(axis=0))[0]]
        raise ValueError(f"Sensor seluruhnya kosong di train: {kosong}")
    imputer = SimpleImputer(strategy="median")
    skala = StandardScaler()
    skala.fit(imputer.fit_transform(train))
    fitur = []
    for split in splits:
        mentah = split[SENSOR].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        fitur.append(skala.transform(imputer.transform(mentah)))
    return (imputer, skala), fitur


def laporkan(metrik: dict) -> str:
    baris = [
        "# Hasil eksperimen pada dataset publik bijih besi",
        "",
        "Metrik dalam poin persentase `% Silica Concentrate` (kecuali R²). Hasil ini bukan performa untuk AMMAN.",
        "",
        "| Model | MAE | RMSE | R² | P90 galat absolut |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for nama, nilai in metrik["test"].items():
        baris.append(f"| {nama} | {nilai['mae']:.4f} | {nilai['rmse']:.4f} | {nilai['r2']:.4f} | {nilai['p90_galat_absolut']:.4f} |")
    baris.extend([
        "", f"Rentang test: {metrik['split']['test']['mulai']} sampai {metrik['split']['test']['akhir']}.",
        "", "Validasi dilakukan kronologis; test hanya digunakan sekali untuk evaluasi akhir.",
        "Periksa latensi lab, drift, dan konsistensi antarperiode sebelum mencoba pilot.",
    ])
    return "\n".join(baris) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/MiningProcess_Flotation_Plant_Database.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs dan batch-size harus positif")
    if not args.csv.is_file():
        parser.error(f"CSV tidak ada: {args.csv}; jalankan python -m src.unduh_data")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (RuntimeError, AttributeError):
        pass

    df = baca_per_jam(args.csv)
    splits = bagi_waktu(df)
    (imputer, skala), fitur = transformasi(splits)
    kumpulan = [jendela(part, x) for part, x in zip(splits, fitur)]
    (x_train, y_train, lab_train, t_train), (x_val, y_val, lab_val, t_val), (x_test, y_test, lab_test, t_test) = kumpulan

    ridge = Ridge(alpha=1.0).fit(x_train[:, -1, :], y_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1], len(SENSOR))),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    riwayat = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    p_ridge = ridge.predict(x_test[:, -1, :])
    p_lstm = model.predict(x_test, verbose=0).reshape(-1)
    prediksi = {"lab_terakhir": lab_test, "ridge": p_ridge, "lstm": p_lstm}
    ringkasan = {
        "sumber": "edumagalhaes/quality-prediction-in-a-mining-process",
        "target": TARGET,
        "fitur": SENSOR,
        "jam_input": 6,
        "horizon_jam": 1,
        "seed": args.seed,
        "epoch_terlatih": len(riwayat.history["loss"]),
        "split": {
            nama: {"mulai": str(part.index.min()), "akhir": str(part.index.max()), "jumlah_jam": len(part), "jumlah_jendela": len(kumpulan[i][1])}
            for i, (nama, part) in enumerate(zip(("train", "validasi", "test"), splits))
        },
        "validasi": {
            "lab_terakhir": skor(y_val, lab_val),
            "ridge": skor(y_val, ridge.predict(x_val[:, -1, :])),
            "lstm": skor(y_val, model.predict(x_val, verbose=0).reshape(-1)),
        },
        "test": {nama: skor(y_test, pred) for nama, pred in prediksi.items()},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "model.keras")
    batas_data = splits[0][SENSOR].replace([np.inf, -np.inf], np.nan)
    joblib.dump({
        "imputer": imputer, "skala": skala, "fitur": SENSOR, "jam_input": 6,
        "batas_bawah": batas_data.quantile(0.005).to_numpy(dtype=float),
        "batas_atas": batas_data.quantile(0.995).to_numpy(dtype=float),
    }, args.out / "preprocessor.joblib")
    joblib.dump(ridge, args.out / "ridge.joblib")
    (args.out / "metrics.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "laporan_hasil.md").write_text(laporkan(ringkasan), encoding="utf-8")
    pd.DataFrame({
        "jam_target": t_test.astype(str), "aktual": y_test,
        "lab_terakhir": lab_test, "ridge": p_ridge, "lstm": p_lstm,
    }).to_csv(args.out / "prediksi_test.csv", index=False)
    print(laporkan(ringkasan))
    print(f"Artefak tersimpan di {args.out}")


if __name__ == "__main__":
    main()
