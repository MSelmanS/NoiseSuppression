"""Temiz konuşma + gürültü -> hedef SNR'da karışım.

Sentetik benchmark'ın çekirdeği: objektif kalite metriklerini (PESQ/STOI/SI-SDR)
hesaplayabilmek için kontrollü, referansı bilinen gürültülü sesler üretiyoruz.
"""

import numpy as np


def pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """arr'ı target_len uzunluğuna getir.

    Daha uzunsa baştan kırp. Daha kısaysa tile et (tekrar ederek doldur).
    Mono (1D) bekleniyor.
    """
    n = len(arr)
    if n >= target_len:
        return arr[:target_len]
    repeats = (target_len + n - 1) // n  # tavan bölme
    tiled = np.tile(arr, repeats)
    return tiled[:target_len]


def extend_to_min_duration(
    audio: np.ndarray,
    sr: int,
    min_seconds: float,
    strategy: str = "mirror",
) -> np.ndarray:
    """Sesi en az min_seconds uzunluğuna getir (zaten yeterliyse aynen döner).

    Strateji:
      "mirror" — audio + audio[::-1] + audio + ... (alternatif ileri/geri)
                 Konuşmacı kimliği ve spektral içerik korunur, STOI/PESQ
                 doğal akışa daha yakın hisseder.
      "tile"   — audio + audio + ... (basit tekrar; sınırda dikiş duyulabilir
                 ama metrikler için yeterli)

    Çıktı uzunluğu tam olarak ceil(min_seconds * sr); ses zaten uzunsa
    olduğu gibi döner (kırpılmaz).
    """
    target_samples = int(np.ceil(min_seconds * sr))
    n = len(audio)
    if n >= target_samples:
        return audio

    parts = []
    total = 0
    flip = False
    while total < target_samples:
        if strategy == "mirror":
            parts.append(audio[::-1] if flip else audio)
            flip = not flip
        else:  # tile
            parts.append(audio)
        total += n

    return np.concatenate(parts)[:target_samples].astype(audio.dtype)


def _rms(x: np.ndarray) -> float:
    """Sinyalin RMS'i. Çok küçük bir floor ekliyoruz ki bölmede patlamasın."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Clean + scale*noise = hedef SNR (dB) karışımı.

    Adımlar:
      1. noise'u clean uzunluğuna pad/trim et
      2. RMS'leri ölç, scale = clean_rms / (noise_rms * 10^(snr_db/20))
      3. mixed = clean + scale * noise
      4. clip [-1, 1]

    Varsayım: clean/noise float32 mono ve aynı sample rate'te.
    Çıktı float32 ve clean ile aynı uzunlukta.

    Not: SNR doğrulaması -- 10*log10(clean_rms^2 / scaled_noise_rms^2)
    tam olarak snr_db'ye eşit olmalı (mixing additif olduğu için).
    """
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    noise = pad_or_trim(noise, len(clean))

    clean_rms = _rms(clean)
    noise_rms = _rms(noise)

    # Hedef SNR formülü: SNR_dB = 20 * log10(clean_rms / (scale * noise_rms))
    # -> scale = clean_rms / (noise_rms * 10^(snr_db/20))
    scale = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))

    mixed = clean + scale * noise
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    return mixed
