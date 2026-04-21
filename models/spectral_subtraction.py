"""Klasik spectral subtraction. Baseline olarak kullanılır, ML içermez.

Mantık: STFT al -> ilk birkaç frame'den gürültü spektrumunu tahmin et
-> her frame'den çıkar -> ISTFT ile zaman domainine geri dön.
"""

import numpy as np
from scipy.signal import stft, istft

from models.base import BaseDenoiser


class SpectralSubtraction(BaseDenoiser):
    name = "spectral_subtraction"
    sample_rate = 16000

    def __init__(self, noise_frames=6, over_subtraction=1.5, floor=0.02):
        # noise_frames     : başlangıçtaki kaç frame gürültü varsayılacak (ilk ~100 ms)
        # over_subtraction : gürültü tahminini biraz abartarak çıkar (müzikal gürültüyü azaltır)
        # floor            : negatife düşmesin diye alt sınır (sinyal spektrumunun oranı olarak)
        self.noise_frames = noise_frames
        self.over_subtraction = over_subtraction
        self.floor = floor

    def load(self):
        # Yüklenecek ağırlık yok; arayüz için boş bırakıyoruz.
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        # STFT: zaman sinyalini zaman-frekans grid'ine çevir.
        # nperseg=512 -> 16 kHz'de ~32 ms pencere (konuşma için standart).
        f, t, Z = stft(audio, fs=self.sample_rate, nperseg=512, noverlap=384)

        # Genlik (magnitude) ve faz ayrı ayrı ele alınır; gürültüyü genlikten çıkarıp fazı koruruz.
        mag = np.abs(Z)
        phase = np.angle(Z)

        # Gürültü spektrumu tahmini: ilk N frame'in ortalaması.
        # Varsayım: kaydın başı sessizdir / sadece gürültü içerir.
        noise_mag = mag[:, :self.noise_frames].mean(axis=1, keepdims=True)

        # Çıkarma. over_subtraction ile biraz fazla çıkar, floor ile negatife düşmeyi engelle.
        clean_mag = mag - self.over_subtraction * noise_mag
        clean_mag = np.maximum(clean_mag, self.floor * mag)

        # Temizlenmiş genlik + orijinal faz -> karmaşık spektrum
        clean_Z = clean_mag * np.exp(1j * phase)

        # ISTFT: zaman domainine geri dön
        _, clean_audio = istft(clean_Z, fs=self.sample_rate, nperseg=512, noverlap=384)

        # STFT/ISTFT uzunluğu birkaç örnek kaydırabilir; orijinale hizala.
        return clean_audio[:len(audio)].astype(np.float32)
