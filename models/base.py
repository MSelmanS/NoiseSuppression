"""Tüm denoiser modellerin uyacağı ortak arayüz.

Her model bu sınıftan türeyecek ve load() + process() metodlarını dolduracak.
Benchmark ve pipeline kodu modelin ne olduğunu bilmeden bu arayüz üzerinden çalışır.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDenoiser(ABC):
    """Soyut temel sınıf. Doğrudan örneklenemez, ondan türemek zorundasın."""

    # Modelin beklediği örnekleme hızı. Alt sınıflar override edebilir (örn. 48000).
    sample_rate = 16000

    # Gösterim/rapor için okunabilir isim. Alt sınıflar override eder.
    name = "base"

    @abstractmethod
    def load(self):
        """Ağırlıkları/kaynakları yükle. Pahalı işlem burada yapılır, process() hızlı olsun."""
        ...

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Gürültülü sesi temizle.

        Girdi  : float32, mono, şekli (N,), sample_rate hızında
        Çıktı  : aynı şekil/format, temizlenmiş ses
        """
        ...
