"""Tek modeli ölçme runner'ı.

İki giriş noktası:
- run_model: model_class verilir, sınıfı örnekler + load + process'leri ölçer.
- run_processing_only: yüklenmiş bir model verilir, sadece process tekrarlarını ölçer.

Sentetik benchmark'ta (model x SNR x pair) load süresi tekrarlamamalı; bu yüzden
load işini script üstleniyor, biz sadece run_processing_only çağırıyoruz.
"""

from __future__ import annotations

import gc
import numpy as np

from .metrics import (
    PeakRSSTracker,
    time_it,
    model_size_mb,
    param_count,
    si_sdr,
    stoi_score,
    pesq_score,
)


def _compute_quality(reference: np.ndarray, estimate: np.ndarray, sr: int) -> dict:
    """Üç kalite metriğini hesapla, isimli dict döndür."""
    return {
        "si_sdr": si_sdr(reference, estimate),
        "stoi": stoi_score(reference, estimate, sr=sr),
        "pesq": pesq_score(reference, estimate, sr=sr),
    }


def run_processing_only(
    model,
    audio: np.ndarray,
    reference: np.ndarray | None = None,
    n_repeats: int = 3,
    do_warmup: bool = True,
) -> tuple[dict, np.ndarray]:
    """Yüklenmiş model üzerinde sadece process()'i ölç.

    AkıŞ:
      1. do_warmup -> process'i bir kez çağır (ölçüm dışı, JIT/cache ısınsın)
      2. n_repeats kez process(audio), her birinin süresini topla
      3. Mean/std hesapla, son çıktıyı geri ver

    Returns:
      result: dict
        - process_time_mean_s, process_time_std_s, rtf_mean, rtf_std,
          audio_duration_s, (varsa) si_sdr/stoi/pesq
      denoised: son koşunun çıktısı
    """
    if do_warmup:
        # Warmup hatası fatal olmasın (örn. çok kısa ses); sadece atla
        try:
            model.process(audio)
        except Exception:
            pass

    times = []
    denoised = None
    for _ in range(n_repeats):
        result, t = time_it(model.process, audio)
        times.append(t)
        denoised = result

    times_arr = np.array(times, dtype=np.float64)
    audio_duration = len(audio) / model.sample_rate
    rtfs = times_arr / audio_duration

    out: dict = {
        "audio_duration_s": round(audio_duration, 3),
        "process_time_mean_s": round(float(times_arr.mean()), 3),
        "process_time_std_s": round(float(times_arr.std(ddof=0)), 3),
        "rtf_mean": round(float(rtfs.mean()), 3),
        "rtf_std": round(float(rtfs.std(ddof=0)), 3),
        "n_repeats": n_repeats,
    }

    if reference is not None and denoised is not None:
        quality = _compute_quality(reference, denoised, sr=model.sample_rate)
        # Yuvarlama: kalite metrikleri 3 basamak yeterli
        out["si_sdr"] = round(quality["si_sdr"], 3)
        out["stoi"] = round(quality["stoi"], 3) if quality["stoi"] == quality["stoi"] else float("nan")
        out["pesq"] = round(quality["pesq"], 3) if quality["pesq"] == quality["pesq"] else float("nan")

    return out, denoised


def run_model(
    model_class,
    audio: np.ndarray,
    reference: np.ndarray | None = None,
    n_repeats: int = 3,
    do_warmup: bool = True,
) -> tuple[dict, np.ndarray]:
    """Tek model için tüm ölçümleri yap (load + process'ler).

    PeakRSSTracker tüm yükleme + process tekrarları boyunca açık kalır, böylece
    geçici RAM patlamaları da yakalanır.

    Returns:
      result: dict
        - model, load_time_s, audio_duration_s
        - process_time_mean_s, process_time_std_s, rtf_mean, rtf_std
        - peak_ram_mb, model_size_mb, param_count
        - si_sdr, stoi, pesq (reference verildiyse)
      denoised: son koşunun çıktısı (.wav olarak kaydetmek için)
    """
    with PeakRSSTracker() as tracker:
        model = model_class()
        _, load_time = time_it(model.load)

        proc_result, denoised = run_processing_only(
            model,
            audio,
            reference=reference,
            n_repeats=n_repeats,
            do_warmup=do_warmup,
        )

    # Model nesnesi üzerinden boyut/param bilgisi (mümkünse PyTorch model'inden)
    inner = model.model if hasattr(model, "model") else None
    size_mb = model_size_mb(inner)
    n_params = param_count(inner)

    result: dict = {
        "model": model.name,
        "load_time_s": round(load_time, 3),
        **proc_result,
        "peak_ram_mb": round(tracker.peak_mb, 1),
        "model_size_mb": round(size_mb, 2),
        "param_count": n_params,
    }

    # Bir sonraki model için RAM'i temizle
    del model
    gc.collect()

    return result, denoised
