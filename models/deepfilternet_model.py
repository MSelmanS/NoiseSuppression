"""DeepFilterNet wrapper. BaseDenoiser arayüzüne uyarlar.

DeepFilterNet 48 kHz'de çalışan bir derin öğrenme modeli (gerçek zamanlı hedefli).
Bizim pipeline 16 kHz kullandığı için ses girişte 48 kHz'e yükseltilir, çıkışta tekrar indirilir.
"""

import numpy as np
import torch
from scipy.signal import resample_poly

from models.base import BaseDenoiser


class DeepFilterNetDenoiser(BaseDenoiser):
    name = "deepfilternet"
    sample_rate = 16000  # Dışarıya 16 kHz; model içeride 48 kHz kullanıyor

    # Modelin kendi iç örnekleme hızı (değişmez, modelin tasarımı böyle)
    _internal_sr = 48000

    def __init__(self):
        self.model = None
        self.df_state = None

    def load(self):
        # Lazy import: kullanıcı bu modeli kullanmayacaksa import gecikmesinden kaçınılır.
        from df.enhance import init_df

        # init_df() modelin ağırlıklarını indirir (ilk çağrıda) ve state döner.
        # model : PyTorch nesnesi, df_state : DSP için gerekli durum bilgisi
        self.model, self.df_state, _ = init_df()

    def process(self, audio: np.ndarray) -> np.ndarray:
        from df.enhance import enhance

        # 16 kHz -> 48 kHz yükselt (modelin beklediği hız)
        audio_48k = resample_poly(audio, up=self._internal_sr, down=self.sample_rate).astype(np.float32)

        # DeepFilterNet torch.Tensor bekliyor, şekil (kanal, N). Mono olduğumuz için (1, N).
        tensor = torch.from_numpy(audio_48k).unsqueeze(0)

        # Gerçek temizleme işlemi
        enhanced = enhance(self.model, self.df_state, tensor)

        # Tekrar NumPy'a dön, kanalı çıkar -> (N,)
        enhanced_np = enhanced.squeeze(0).cpu().numpy()

        # 48 kHz -> 16 kHz indir (pipeline'ın beklediği format)
        out = resample_poly(enhanced_np, up=self.sample_rate, down=self._internal_sr).astype(np.float32)

        # Resample sonucu bazen 1-2 sample kayabilir; uzunluğu hizala
        return out[:len(audio)]
