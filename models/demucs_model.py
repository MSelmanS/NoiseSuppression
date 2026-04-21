"""Facebook Denoiser (Demucs mimarisi) wrapper.

Üç varyantı destekler: dns48 (hafif), dns64 (orta), master64 (en kaliteli).
Alt sınıflar sadece variant seçer.
"""

import numpy as np
import torch

from models.base import BaseDenoiser


class _DemucsBase(BaseDenoiser):
    """Ortak Demucs mantığı. Alt sınıflar 'variant' attribute'unu set eder."""

    sample_rate = 16000
    variant = None   # alt sınıflar 'dns48', 'dns64' veya 'master64' verir

    def __init__(self):
        self.model = None

    def load(self):
        # Lazy import + variant'a göre doğru factory fonksiyonunu seç
        from denoiser import pretrained

        # getattr(modül, isim) -> modül içindeki fonksiyonu ismiyle getirir
        # Yani variant="dns48" ise pretrained.dns48'i çağırmış oluruz
        factory = getattr(pretrained, self.variant)
        self.model = factory()
        self.model.eval()

    def process(self, audio: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        out_np = out.squeeze().cpu().numpy()
        return out_np[:len(audio)].astype(np.float32)


class DemucsDns48(_DemucsBase):
    name = "demucs_dns48"
    variant = "dns48"


class DemucsDns64(_DemucsBase):
    name = "demucs_dns64"
    variant = "dns64"


class DemucsMaster64(_DemucsBase):
    name = "demucs_master64"
    variant = "master64"
