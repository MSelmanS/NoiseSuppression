"""Bench profilleri.

Senaryo S (kanonik koşum) ve smoke testler için tek yerde tutulan parametre
setleri. `bench_synthetic.py --profile NAME` ile çağrılır; ayrı CLI argümanları
profile değerlerini override eder.

Yeni profil ekleme şablonu:

    "yeni_profil": {
        "clean_dir": "data/clean",
        "noise_dir": "data/noise",
        "snrs": [-5, 0, 5, 10, 15],
        "max_pairs": 20,
        "n_repeats": 3,
        "models": "all",
        "description": "Bir cümleyle profilin amacı."
    },
"""

from __future__ import annotations


PROFILES: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Senaryo S — kanonik koşum
    # 20 temiz × 7 noise × 5 SNR × 3 rep = 2100 işlem/model × 7 model
    # CPU'da tahmini süre: 30-60 dk
    # -----------------------------------------------------------------------
    "s_quick": {
        "clean_dir": "data/clean",
        "noise_dir": "data/noise",
        "snrs": [-5.0, 0.0, 5.0, 10.0, 15.0],
        "max_pairs": 20,
        "n_repeats": 3,
        "models": "all",
        "description": "Senaryo S: 20×7×5×3 = 2100 ölçüm/model, ~30-60dk CPU.",
    },

    # -----------------------------------------------------------------------
    # Smoke test — hızlı sağlık kontrolü
    # 4 temiz × 2 noise × 2 SNR × 1 rep, sadece hızlı modeller
    # Hedef: 5 dakikadan kısa
    # -----------------------------------------------------------------------
    "s_smoke": {
        "clean_dir": "data/clean",
        "noise_dir": "data/noise",
        "snrs": [0.0, 10.0],
        "max_pairs": 4,
        "n_repeats": 1,
        "models": "spectral_subtraction,rnnoise,deepfilternet,metricgan_plus",
        "description": "Smoke: 4×2×2×1 hızlı modeller, <5dk.",
    },

    # -----------------------------------------------------------------------
    # Orta senaryo M — Senaryo S'in genişletilmişi
    # 50 temiz × 7 noise × 5 SNR × 3 rep = 5250 ölçüm/model — saatlerce sürer
    # Sonraki turlarda gerekirse açılır.
    # -----------------------------------------------------------------------
    "m_medium": {
        "clean_dir": "data/clean",
        "noise_dir": "data/noise",
        "snrs": [-5.0, 0.0, 5.0, 10.0, 15.0],
        "max_pairs": 50,
        "n_repeats": 3,
        "models": "all",
        "description": "Senaryo M: 50×7×5×3, saatler sürebilir.",
    },
}


def estimate_measurements(profile: dict, n_models: int = 7) -> int:
    """Toplam process() çağrı sayısını kaba tahmin et (warmup hariç)."""
    n_pairs = profile["max_pairs"]
    n_snrs = len(profile["snrs"])
    n_reps = profile["n_repeats"]
    return n_pairs * n_snrs * n_reps * n_models


def get_profile(name: str) -> dict:
    """İsim ile profil getir; bilinmeyen isimde anlamlı hata fırlat."""
    if name not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise ValueError(
            f"Bilinmeyen profil: '{name}'. Geçerli profiller: {valid}"
        )
    return PROFILES[name]
