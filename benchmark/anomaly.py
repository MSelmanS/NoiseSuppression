"""Benchmark sonuçları üzerinde anomali yakalama.

Dört kural (bkz. development_task.md Task 4):
  1. PESQ-RMS çelişkisi:    PESQ yüksek ama output RMS noisy'den çok düşük
  2. Yüksek varyans:        Aynı (model, scene, snr) için PESQ std > 0.5
  3. SNR trend reversal:    SNR artarken PESQ düşüyor
  4. Aşırı bastırma:        HF Ratio < 0.001 veya output RMS << noisy RMS

Çıktı: her anomalili satır için bir kayıt (model, scene, snr, type, severity,
details). Eşikler kalibrasyon kolaylığı için fonksiyon başında sabit.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


# Eşikler (gelecekte kalibrasyon kolay olsun diye burada)
PESQ_OUTLIER_OFFSET = 0.3       # mean(PESQ) + bu kadar yüksekse "yüksek skor"
PESQ_RMS_DROP_DB = 6.0          # output RMS noisy'den bu kadar düşükse "kırpılmış"
PESQ_STD_THRESHOLD = 0.5        # PESQ std bu üzerindeyse tutarsız
OVER_SUPPRESS_RMS_DB = 10.0     # output RMS noisy'den bu kadar düşükse aşırı bastırma
HF_RATIO_THRESHOLD = 0.0010     # HF ratio bu altındaysa aşırı bastırma


def _scene_from_noise(noise_file: str) -> str:
    """data/noise/SCAFE/ch01.wav -> SCAFE veya basename'in parent'ı."""
    if not noise_file:
        return "unknown"
    # noise_file 'noise_file' kolonunda sadece basename veya tam yol olabilir.
    parts = os.path.normpath(noise_file).split(os.sep)
    if len(parts) >= 2 and parts[-1].endswith(".wav"):
        return parts[-2]
    return parts[-1].replace(".wav", "")


def _row_dict(row: pd.Series, anomaly_type: str, severity: str, details: str) -> dict[str, Any]:
    return {
        "model": row.get("model", ""),
        "scene": _scene_from_noise(str(row.get("noise_file", ""))),
        "snr_db": row.get("snr_db"),
        "clean_file": row.get("clean_file", ""),
        "noise_file": row.get("noise_file", ""),
        "type": anomaly_type,
        "severity": severity,
        "details": details,
        # Referans için ilgili sayısal alanlar (yorumlanabilirlik)
        "pesq": row.get("pesq"),
        "stoi": row.get("stoi"),
        "si_sdr": row.get("si_sdr"),
        "noisy_rms_db": row.get("noisy_rms_db"),
        "output_rms_db": row.get("output_rms_db"),
        "hf_ratio": row.get("hf_ratio"),
    }


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Sonuç DataFrame'ini incele, anomalileri DataFrame olarak döndür.

    Beklenen kolonlar (varsa kullanılır, yoksa o kural atlanır):
      model, snr_db, noise_file, pesq, stoi, si_sdr,
      noisy_rms_db, output_rms_db, hf_ratio
    """
    anomalies: list[dict[str, Any]] = []
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["model", "scene", "snr_db", "type", "severity", "details"])

    # --- Rule 1: PESQ-RMS mismatch ---------------------------------------
    has_pesq = "pesq" in df.columns and df["pesq"].notna().any()
    has_rms = "noisy_rms_db" in df.columns and "output_rms_db" in df.columns
    if has_pesq and has_rms:
        pesq_mean = float(df["pesq"].mean(skipna=True))
        threshold = pesq_mean + PESQ_OUTLIER_OFFSET
        for _, row in df.iterrows():
            pesq = row.get("pesq")
            if pd.isna(pesq) or pesq < threshold:
                continue
            n_rms = row.get("noisy_rms_db")
            o_rms = row.get("output_rms_db")
            if pd.isna(n_rms) or pd.isna(o_rms):
                continue
            drop = float(n_rms) - float(o_rms)
            if drop >= PESQ_RMS_DROP_DB:
                anomalies.append(_row_dict(
                    row,
                    "pesq_rms_mismatch",
                    "high",
                    f"PESQ={pesq:.2f} >= ortalama+{PESQ_OUTLIER_OFFSET} ({threshold:.2f}) "
                    f"ama output RMS noisy'den {drop:.1f} dB düşük "
                    f"(noisy={n_rms:.1f} dB, output={o_rms:.1f} dB)",
                ))

    # --- Rule 2: High variance (same model/scene/snr) --------------------
    if has_pesq and "noise_file" in df.columns and "snr_db" in df.columns:
        grouped = df.groupby(["model", "noise_file", "snr_db"], dropna=False)
        for (model, noise_file, snr), grp in grouped:
            pesq_values = grp["pesq"].dropna()
            if len(pesq_values) < 2:
                continue
            std = float(pesq_values.std(ddof=0))
            if std > PESQ_STD_THRESHOLD:
                row = grp.iloc[0].copy()
                anomalies.append(_row_dict(
                    row,
                    "high_variance",
                    "medium",
                    f"({model}, {_scene_from_noise(str(noise_file))}, snr={snr}) PESQ std={std:.2f} > {PESQ_STD_THRESHOLD}",
                ))

    # --- Rule 3: SNR trend reversal --------------------------------------
    # SNR arttıkça PESQ ortalaması artmalı; iki ardışık SNR'da PESQ düşüyorsa anomali.
    if has_pesq and "snr_db" in df.columns:
        keys = [c for c in ("model", "noise_file") if c in df.columns]
        if keys:
            for key_vals, grp in df.groupby(keys, dropna=False):
                agg = grp.groupby("snr_db")["pesq"].mean().sort_index()
                snrs = agg.index.tolist()
                pesqs = agg.values.tolist()
                for i in range(1, len(pesqs)):
                    a = pesqs[i - 1]
                    b = pesqs[i]
                    if pd.isna(a) or pd.isna(b):
                        continue
                    if b < a - 0.05:  # 0.05 PESQ kayma toleransı
                        # Temsili satır olarak grup ilk satırı
                        row = grp.iloc[0].copy()
                        # key_vals tuple veya tek değer olabilir
                        kv = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                        ctx = ", ".join(f"{k}={v}" for k, v in zip(keys, kv))
                        anomalies.append(_row_dict(
                            row,
                            "snr_trend_reversal",
                            "low",
                            f"({ctx}) SNR {snrs[i-1]} -> {snrs[i]} dB'de PESQ {a:.2f} -> {b:.2f}",
                        ))

    # --- Rule 4: Over-suppression (HF ratio veya RMS düşüşü) -------------
    has_hf = "hf_ratio" in df.columns and df["hf_ratio"].notna().any()
    for _, row in df.iterrows():
        details_parts: list[str] = []
        hf = row.get("hf_ratio")
        if has_hf and pd.notna(hf) and hf < HF_RATIO_THRESHOLD:
            details_parts.append(f"HF ratio={hf:.5f} < {HF_RATIO_THRESHOLD}")
        if has_rms:
            n_rms = row.get("noisy_rms_db")
            o_rms = row.get("output_rms_db")
            if pd.notna(n_rms) and pd.notna(o_rms):
                drop = float(n_rms) - float(o_rms)
                if drop >= OVER_SUPPRESS_RMS_DB:
                    details_parts.append(
                        f"RMS düşüşü {drop:.1f} dB >= {OVER_SUPPRESS_RMS_DB} dB"
                    )
        if details_parts:
            anomalies.append(_row_dict(
                row,
                "over_suppression",
                "high",
                "; ".join(details_parts),
            ))

    return pd.DataFrame(anomalies)


def summarize_anomalies(anomalies: pd.DataFrame) -> dict[str, int]:
    """Tür başına anomali sayısı; konsola hızlı bakış için."""
    if anomalies is None or len(anomalies) == 0:
        return {}
    return anomalies["type"].value_counts().to_dict()
