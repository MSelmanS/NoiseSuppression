"""Önceden kayıt altına alınan H1-H4 hipotezlerinin otomatik testi.

Hipotezler `development_task.md` § 1.4'te tanımlı. Bu modül koşum sonuçlarını
(DataFrame) okur, her hipotez için VERIFIED / PARTIAL / REJECTED kararı
+ dayanak rakamlar üretir.

Bölüm 1.4'ün özeti:
  H1. SNR <= 0 dB'de klasik (spectral_subtraction + rnnoise) PESQ'si
      DL'den (DFN + Demucs* + MetricGAN+) >= 0.3 düşüktür.
  H2. Sabit gürültü sahnelerinde (OOFFICE, TCAR, NPARK; eldeki DEMAND'da
      DKITCHEN yok) DFN + Demucs PESQ'si klasikten >= 0.4 yüksektir.
  H3. MetricGAN+ çıkışlarında en az 2 SNR'da RMS düşüşü > 8 dB veya
      HF Ratio < 0.0010 (aşırı agresiflik göstergesi).
  H4. SCAFE + SPSQUARE'da modellerin PESQ ortalama farkı (max-min) < 0.4.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# Model grupları
CLASSIC_MODELS = ("spectral_subtraction", "rnnoise")
DL_MODELS = ("deepfilternet", "demucs_dns48", "demucs_dns64", "demucs_master64", "metricgan_plus")

# Sahne grupları (DEMAND alt küme isimleri)
STATIONARY_SCENES = ("OOFFICE", "TCAR", "NPARK")
CROWD_SCENES = ("SCAFE", "SPSQUARE")

# Hipotez eşikleri
H1_PESQ_GAP = 0.3
H2_PESQ_GAP = 0.4
H3_RMS_DROP_DB = 8.0
H3_HF_RATIO = 0.0010
H4_PESQ_SPREAD = 0.4


def _mean_safe(s: pd.Series) -> float:
    """NaN-safe ortalama; tüm NaN ise nan döner."""
    return float(s.mean(skipna=True)) if len(s) else float("nan")


def _status(verified: bool, partial: bool = False) -> str:
    if verified:
        return "VERIFIED"
    if partial:
        return "PARTIAL"
    return "REJECTED"


def test_h1(df: pd.DataFrame) -> dict[str, Any]:
    """SNR <= 0 dB'de klasik PESQ << DL PESQ."""
    if "pesq" not in df.columns or "snr_db" not in df.columns:
        return {"id": "H1", "status": "INSUFFICIENT_DATA", "evidence": {}}
    low_snr = df[df["snr_db"] <= 0]
    if len(low_snr) == 0:
        return {"id": "H1", "status": "INSUFFICIENT_DATA", "evidence": {"reason": "SNR <= 0 satır yok"}}
    classic = low_snr[low_snr["model"].isin(CLASSIC_MODELS)]["pesq"]
    dl = low_snr[low_snr["model"].isin(DL_MODELS)]["pesq"]
    classic_mean = _mean_safe(classic)
    dl_mean = _mean_safe(dl)
    gap = dl_mean - classic_mean
    verified = gap >= H1_PESQ_GAP
    return {
        "id": "H1",
        "title": "Klasik modeller (SS, RNNoise) düşük SNR'da DL'den geride",
        "status": _status(verified),
        "evidence": {
            "classic_pesq_mean": round(classic_mean, 3),
            "dl_pesq_mean": round(dl_mean, 3),
            "gap": round(gap, 3),
            "threshold": H1_PESQ_GAP,
            "n_classic_rows": int(classic.notna().sum()),
            "n_dl_rows": int(dl.notna().sum()),
        },
    }


def test_h2(df: pd.DataFrame) -> dict[str, Any]:
    """Sabit gürültü sahnelerinde DFN + Demucs >> klasik."""
    if "pesq" not in df.columns or "scene" not in df.columns:
        return {"id": "H2", "status": "INSUFFICIENT_DATA", "evidence": {}}
    stationary = df[df["scene"].isin(STATIONARY_SCENES)]
    if len(stationary) == 0:
        return {"id": "H2", "status": "INSUFFICIENT_DATA",
                "evidence": {"reason": f"Sabit sahneler yok: {STATIONARY_SCENES}"}}
    # DFN + Demucs alt-grubu (MetricGAN+ hariç, bu hipotezde sadece konvansiyonel DL'ler)
    strong_dl = ("deepfilternet", "demucs_dns48", "demucs_dns64", "demucs_master64")
    dl_mean = _mean_safe(stationary[stationary["model"].isin(strong_dl)]["pesq"])
    classic_mean = _mean_safe(stationary[stationary["model"].isin(CLASSIC_MODELS)]["pesq"])
    gap = dl_mean - classic_mean
    verified = gap >= H2_PESQ_GAP
    return {
        "id": "H2",
        "title": "Sabit gürültü sahnelerinde DFN + Demucs >> klasik",
        "status": _status(verified, partial=(gap > 0 and gap < H2_PESQ_GAP)),
        "evidence": {
            "scenes": list(STATIONARY_SCENES),
            "dl_pesq_mean": round(dl_mean, 3),
            "classic_pesq_mean": round(classic_mean, 3),
            "gap": round(gap, 3),
            "threshold": H2_PESQ_GAP,
        },
    }


def test_h3(df: pd.DataFrame) -> dict[str, Any]:
    """MetricGAN+ aşırı agresif: >= 2 SNR'da RMS düşüşü > 8 dB veya HF < 0.001."""
    if df is None or len(df) == 0 or "model" not in df.columns:
        return {"id": "H3", "status": "INSUFFICIENT_DATA", "evidence": {}}
    mg = df[df["model"] == "metricgan_plus"]
    if len(mg) == 0:
        return {"id": "H3", "status": "INSUFFICIENT_DATA",
                "evidence": {"reason": "MetricGAN+ satırı yok"}}

    aggressive_snrs = []
    for snr, grp in mg.groupby("snr_db"):
        flag_reasons = []
        if "noisy_rms_db" in grp.columns and "output_rms_db" in grp.columns:
            drops = grp["noisy_rms_db"] - grp["output_rms_db"]
            mean_drop = float(drops.mean(skipna=True))
            if mean_drop > H3_RMS_DROP_DB:
                flag_reasons.append(f"mean_rms_drop={mean_drop:.1f}dB")
        if "hf_ratio" in grp.columns:
            mean_hf = float(grp["hf_ratio"].mean(skipna=True))
            if mean_hf < H3_HF_RATIO:
                flag_reasons.append(f"mean_hf={mean_hf:.5f}")
        if flag_reasons:
            aggressive_snrs.append({"snr": snr, "reasons": flag_reasons})

    verified = len(aggressive_snrs) >= 2
    return {
        "id": "H3",
        "title": "MetricGAN+ en az 2 SNR'da aşırı agresif",
        "status": _status(verified, partial=(len(aggressive_snrs) == 1)),
        "evidence": {
            "aggressive_snrs": aggressive_snrs,
            "n_flagged_snrs": len(aggressive_snrs),
            "thresholds": {"rms_drop_db": H3_RMS_DROP_DB, "hf_ratio": H3_HF_RATIO},
        },
    }


def test_h4(df: pd.DataFrame) -> dict[str, Any]:
    """Kalabalık sahnelerde model farkları azalır: max-min PESQ < 0.4."""
    if "pesq" not in df.columns or "scene" not in df.columns:
        return {"id": "H4", "status": "INSUFFICIENT_DATA", "evidence": {}}
    crowd = df[df["scene"].isin(CROWD_SCENES)]
    if len(crowd) == 0:
        return {"id": "H4", "status": "INSUFFICIENT_DATA",
                "evidence": {"reason": f"Kalabalık sahneler yok: {CROWD_SCENES}"}}
    by_model = crowd.groupby("model")["pesq"].mean(skipna=True)
    if len(by_model) < 2:
        return {"id": "H4", "status": "INSUFFICIENT_DATA",
                "evidence": {"reason": "En az 2 model lazım"}}
    spread = float(by_model.max() - by_model.min())
    verified = spread < H4_PESQ_SPREAD
    return {
        "id": "H4",
        "title": "Kalabalık sahnelerde model farkı azalır",
        "status": _status(verified),
        "evidence": {
            "scenes": list(CROWD_SCENES),
            "model_pesq_means": {m: round(v, 3) for m, v in by_model.items()},
            "spread_max_minus_min": round(spread, 3),
            "threshold": H4_PESQ_SPREAD,
        },
    }


def run_all(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Dört hipotezi sırayla test et."""
    return [test_h1(df), test_h2(df), test_h3(df), test_h4(df)]
