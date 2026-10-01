# Hasil eksperimen pada dataset publik bijih besi

Metrik dalam poin persentase `% Silica Concentrate` (kecuali R²). Hasil ini bukan performa untuk AMMAN.

| Model | MAE | RMSE | R² | P90 galat absolut |
| --- | ---: | ---: | ---: | ---: |
| lab_terakhir | 0.4755 | 0.7638 | 0.5953 | 1.2160 |
| ridge | 1.0429 | 1.3711 | -0.3041 | 2.4150 |
| lstm | 1.0452 | 1.3723 | -0.3064 | 2.4513 |

Rentang test: 2017-08-15 09:00:00 sampai 2017-09-09 23:00:00.

Validasi dilakukan kronologis; test hanya digunakan sekali untuk evaluasi akhir.
Periksa latensi lab, drift, dan konsistensi antarperiode sebelum mencoba pilot.
