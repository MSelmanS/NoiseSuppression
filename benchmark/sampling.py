"""Dinleme galerisi için akıllı örnek seçici.

Senaryo S 14.700 wav üretebilir; bunların hepsini diske yazmak gereksiz.
Rapor için en bilgilendirici 28 örneği önceden seçeriz; sadece bunlar kaydedilir.

Seçim kriteri (deterministik, seed=42):
  - Kritik sahneler: SCAFE, TCAR  (kalabalık konuşma + araç içi — hipotezdeki
    iki uç senaryo)
  - Kritik SNR'lar: -5 dB, +5 dB  (en zor + orta zorluk)
  - Her (sahne, snr) için 1 pair seçilir
  - Her model bu pair'lar için ayrı ses üretir
  -> 2 sahne × 2 snr × 7 model = 28 ses dosyası

Ek olarak input dosyaları (model-bağımsız) bir kez kaydedilir:
  - 1 clean reference (pair başına)
  - 4 noisy mix (2 sahne × 2 snr)
"""

from __future__ import annotations

import random
from typing import Iterable


# Hipotezdeki en bilgilendirici sahneler (DEMAND alt küme isimleriyle eşleşmeli)
CRITICAL_SCENES: tuple[str, ...] = ("SCAFE", "TCAR")
# En zor ve orta zorluk
CRITICAL_SNRS: tuple[float, ...] = (-5.0, 5.0)


def pick_listening_samples(
    pairs: list[tuple[str, str]],
    snrs: Iterable[float],
    seed: int = 42,
    critical_scenes: tuple[str, ...] = CRITICAL_SCENES,
    critical_snrs: tuple[float, ...] = CRITICAL_SNRS,
) -> set[tuple[int, float]]:
    """Kaydedilecek (pair_index, snr) kombinasyonlarını seç.

    Args:
      pairs: bench_synthetic'in _sample_pairs çıktısı, [(clean_path, noise_path), ...]
      snrs: koşumdaki tüm SNR değerleri (kontrol için; bu fonksiyon kritik olanları seçer)
      seed: deterministiklik için
      critical_scenes: kaydedilecek sahne isimleri (DEMAND klasörü adıyla eşleşir)
      critical_snrs: kaydedilecek SNR değerleri (yaklaşık eşleşme yapar)

    Returns:
      {(pair_index, snr_value), ...} kümesi. bench_synthetic döngüsünde bu kümede
      olan (pi, snr) için wav kaydedilir, diğerleri için yazılmaz.
    """
    rng = random.Random(seed)
    snrs_list = list(snrs)

    # Hedef SNR'larla mevcut SNR'ları yaklaşık eşle (0.1 dB toleransı)
    matched_snrs: list[float] = []
    for target in critical_snrs:
        match = next((s for s in snrs_list if abs(float(s) - target) < 0.1), None)
        if match is not None:
            matched_snrs.append(match)

    selected: set[tuple[int, float]] = set()
    for scene in critical_scenes:
        # Bu sahneye ait pair'ların indekslerini bul (noise_path scene adını içeriyor mu?)
        scene_pair_indices = [
            i for i, (_clean, noise) in enumerate(pairs)
            if scene.lower() in noise.lower()
        ]
        if not scene_pair_indices:
            # Bu sahne için pair yok; atla
            continue
        # Deterministik 1 pair seç
        chosen_pi = rng.choice(scene_pair_indices)
        for snr in matched_snrs:
            selected.add((chosen_pi, snr))

    return selected


def is_selected(
    pair_index: int,
    snr: float,
    selected: set[tuple[int, float]],
) -> bool:
    """selected içinde (pair_index, ~snr) çiftinin olup olmadığını döndür.

    SNR float karşılaştırması için yaklaşık eşleşme (0.01 dB tol).
    """
    for pi, s in selected:
        if pi == pair_index and abs(s - snr) < 0.01:
            return True
    return False
