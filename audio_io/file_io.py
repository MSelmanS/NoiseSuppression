"""WAV dosyası okuma/yazma. Her çıktı mono + hedef örnekleme hızında olur."""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import MODEL_SR


def load_audio(path, target_sr=MODEL_SR):
    """WAV dosyasını oku; mono ve target_sr olarak döndür.

    Dönüş: (samples, sr)
      samples : np.ndarray, float32, şekli (N,), değerler [-1, 1] aralığında
      sr      : int, örnekleme hızı (target_sr)
    """
    # always_2d=True -> her zaman (N, kanal) şekli; mono'da da (N, 1) olur. İşi tekdüze yapar.
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Çok kanallıysa kanalların ortalamasıyla mono'ya indir
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Örnekleme hızı farklıysa yeniden örnekle. resample_poly kaliteli ve hızlıdır.
    if sr != target_sr:
        data = resample_poly(data, up=target_sr, down=sr).astype(np.float32)
        sr = target_sr

    return data, sr


def save_audio(path, samples, sr=MODEL_SR):
    """Mono float32 örnekleri 16-bit PCM WAV olarak kaydet."""
    # Model çıkışı bazen [-1, 1] dışına taşabilir; kırpmazsak WAV'da bozuk ses olur.
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(path, samples, sr, subtype="PCM_16")
