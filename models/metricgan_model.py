"""MetricGAN+ wrapper (SpeechBrain).

MetricGAN+ bir GAN tabanlı enhancer; PESQ metriğini doğrudan optimize ederek
eğitildi. 16 kHz'de çalışır — bizim pipeline ile birebir uyumlu.
"""

import numpy as np
import torch

from models.base import BaseDenoiser


class MetricGANDenoiser(BaseDenoiser):
    name = "metricgan_plus"
    sample_rate = 16000   # SpeechBrain modeli 16 kHz'de eğitildi

    def __init__(self):
        self.model = None

    def load(self):
        # Lazy import — SpeechBrain ağır bir paket
        # SpeechBrain 0.5.x'te bu sınıf speechbrain.pretrained altında
        # (1.x'te speechbrain.inference.enhancement olarak yeniden adlandırıldı)
        from speechbrain.pretrained import SpectralMaskEnhancement

        # from_hparams HuggingFace'ten modeli indirir ve cache'ler.
        # savedir yerel önbellek klasörü; aynı modele tekrar çağrı yaparsan buradan okur.
        self.model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
        )

    def process(self, audio: np.ndarray) -> np.ndarray:
        # 1) NumPy -> Tensor, (N,) -> (1, N) batch boyutu ekle
        tensor = torch.from_numpy(audio).unsqueeze(0)

        # 2) lengths: batch'teki her örneğin oransal uzunluğu (0-1).
        #    Tek örnek ve padding yok -> [1.0] = "tamamı gerçek veri"
        lengths = torch.tensor([1.0])

        # 3) Inference. SpeechBrain içeride no_grad ve eval() hallediyor, biz eklemiyoruz.
        enhanced = self.model.enhance_batch(tensor, lengths=lengths)

        # 4) (1, N) -> (N,) ve NumPy'a geri dön
        out = enhanced.squeeze(0).cpu().numpy()

        return out[:len(audio)].astype(np.float32)