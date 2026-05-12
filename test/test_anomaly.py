"""Anomali yakalama birim testleri.

Çalıştırma: `python -m test.test_anomaly` (pytest gerekmez).
"""

import pandas as pd

from benchmark.anomaly import detect_anomalies, summarize_anomalies


def _mock_df_pesq_rms_mismatch() -> pd.DataFrame:
    """Bir satır kasten PESQ yüksek ama output RMS çok düşük (anomali olmalı).

    Diğer satırlar normal; ortalama PESQ ~2.5, mismatch satırı PESQ=3.5
    (mean+0.3=2.8'den yüksek) ve output RMS noisy'den 10 dB düşük.
    """
    rows = [
        # Normal satırlar
        {"model": "deepfilternet", "snr_db": 0, "noise_file": "TCAR/ch01.wav",
         "pesq": 2.4, "stoi": 0.6, "si_sdr": 5.0,
         "noisy_rms_db": -20.0, "output_rms_db": -18.0, "hf_ratio": 0.5},
        {"model": "deepfilternet", "snr_db": 5, "noise_file": "TCAR/ch01.wav",
         "pesq": 2.6, "stoi": 0.7, "si_sdr": 8.0,
         "noisy_rms_db": -18.0, "output_rms_db": -17.0, "hf_ratio": 0.6},
        # Anomali: yüksek PESQ + büyük RMS düşüşü
        {"model": "metricgan_plus", "snr_db": 0, "noise_file": "SCAFE/ch01.wav",
         "pesq": 3.5, "stoi": 0.6, "si_sdr": 4.0,
         "noisy_rms_db": -15.0, "output_rms_db": -27.0, "hf_ratio": 0.4},
    ]
    return pd.DataFrame(rows)


def _mock_df_over_suppression() -> pd.DataFrame:
    rows = [
        {"model": "metricgan_plus", "snr_db": -5, "noise_file": "OOFFICE/ch01.wav",
         "pesq": 1.2, "stoi": 0.4, "si_sdr": -2.0,
         "noisy_rms_db": -12.0, "output_rms_db": -25.0, "hf_ratio": 0.0005},
    ]
    return pd.DataFrame(rows)


def _mock_df_high_variance() -> pd.DataFrame:
    # Aynı (model, noise, snr) için 3 farklı pair -> çok yüksek PESQ varyansı
    rows = [
        {"model": "rnnoise", "snr_db": 0, "noise_file": "SPSQUARE/ch01.wav",
         "pesq": 1.0, "noisy_rms_db": -20.0, "output_rms_db": -19.0, "hf_ratio": 0.3},
        {"model": "rnnoise", "snr_db": 0, "noise_file": "SPSQUARE/ch01.wav",
         "pesq": 2.5, "noisy_rms_db": -20.0, "output_rms_db": -19.0, "hf_ratio": 0.3},
        {"model": "rnnoise", "snr_db": 0, "noise_file": "SPSQUARE/ch01.wav",
         "pesq": 3.0, "noisy_rms_db": -20.0, "output_rms_db": -19.0, "hf_ratio": 0.3},
    ]
    return pd.DataFrame(rows)


def _mock_df_trend_reversal() -> pd.DataFrame:
    # SNR arttıkça PESQ düşüyor (anormal)
    rows = [
        {"model": "spectral_subtraction", "snr_db": 0, "noise_file": "NPARK/ch01.wav",
         "pesq": 3.0, "noisy_rms_db": -18.0, "output_rms_db": -18.0, "hf_ratio": 0.5},
        {"model": "spectral_subtraction", "snr_db": 5, "noise_file": "NPARK/ch01.wav",
         "pesq": 2.0, "noisy_rms_db": -16.0, "output_rms_db": -16.0, "hf_ratio": 0.5},
        {"model": "spectral_subtraction", "snr_db": 10, "noise_file": "NPARK/ch01.wav",
         "pesq": 1.5, "noisy_rms_db": -14.0, "output_rms_db": -14.0, "hf_ratio": 0.5},
    ]
    return pd.DataFrame(rows)


def run_all() -> int:
    failures = 0

    df = _mock_df_pesq_rms_mismatch()
    anomalies = detect_anomalies(df)
    print(f"[test 1] pesq_rms_mismatch: {len(anomalies)} anomali")
    print(anomalies[["model", "type", "details"]].to_string(index=False))
    if not any(anomalies["type"] == "pesq_rms_mismatch"):
        print("  [FAIL] pesq_rms_mismatch tipinde anomali yakalanmadı")
        failures += 1

    df = _mock_df_over_suppression()
    anomalies = detect_anomalies(df)
    print(f"\n[test 2] over_suppression: {len(anomalies)} anomali")
    if not any(anomalies["type"] == "over_suppression"):
        print("  [FAIL] over_suppression tipinde anomali yakalanmadı")
        failures += 1

    df = _mock_df_high_variance()
    anomalies = detect_anomalies(df)
    print(f"\n[test 3] high_variance: {len(anomalies)} anomali")
    if not any(anomalies["type"] == "high_variance"):
        print("  [FAIL] high_variance tipinde anomali yakalanmadı")
        failures += 1

    df = _mock_df_trend_reversal()
    anomalies = detect_anomalies(df)
    print(f"\n[test 4] snr_trend_reversal: {len(anomalies)} anomali")
    if not any(anomalies["type"] == "snr_trend_reversal"):
        print("  [FAIL] snr_trend_reversal tipinde anomali yakalanmadı")
        failures += 1

    print(f"\n=== {4 - failures}/4 test başarılı ===")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_all())
