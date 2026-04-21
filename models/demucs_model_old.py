"""Facebook Denoiser (Demucs mimarisi) wrapper.

Model 16 kHz'de eğitilmiş, bizim pipeline da 16 kHz — resample gerekmez.
Giriş/çıkış tensor dönüşümlerini burada yönetiyoruz.
"""

import numpy as np
import torch

from models.base import BaseDenoiser


class DemucsDenoiser(BaseDenoiser):
    name = "demucs_dns48"
    sample_rate = 16000   # Model 16 kHz'de eğitildi, bizim pipeline da öyle

    def __init__(self):
        self.model = None

    def load(self):
        # Lazy import — denoiser pahalı bir paket, sadece kullanılacaksa yüklensin
        from denoiser.pretrained import dns48

        # dns48: hidden=48 olan real-time varyant. İlk çağrıda ağırlıkları
        # torch.hub üzerinden indirir (~20 MB), sonraki çağrılarda cache'ten okur.
        self.model = dns48()

        # Eval moduna al: dropout/batchnorm inference davranışına geçer.
        # Inference'ta ZORUNLU — atlarsan çıktı her seferinde farklı olabilir.
        self.model.eval()

    def process(self, audio: np.ndarray) -> np.ndarray:
        # 1) NumPy -> Tensor
        tensor = torch.from_numpy(audio)

        # 2) Şekli (N,) -> (1 batch, 1 kanal, N)
        #    Model (batch, kanal, N) bekliyor, tek örnek olsa bile batch=1 lazım
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # 3) no_grad: inference'ta autograd kapalı olmalı (hafıza + hız)
        with torch.no_grad():
            out = self.model(tensor)   # şekil: (1, 1, N)

        # 4) Boyutları geri topla: (1, 1, N) -> (N,)
        #    İki kere squeeze; dim belirtmezsek tüm 1'lik boyutlar silinir
        out_np = out.squeeze().cpu().numpy()

        # 5) Uzunluk hizalaması — model 1-2 sample kaydırabilir
        return out_np[:len(audio)].astype(np.float32)
