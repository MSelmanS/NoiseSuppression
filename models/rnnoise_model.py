"""RNNoise wrapper. BaseDenoiser arayüzüne uyarlar.

RNNoise 48 kHz + int16 + (kanal, sample) 2D array ister.
Biz dışarıya 16 kHz float32 1D sunuyoruz; dönüşümler içeride.
"""

import numpy as np
from scipy.signal import resample_poly

from models.base import BaseDenoiser


class RNNoiseDenoiser(BaseDenoiser):
    name = "rnnoise"
    sample_rate = 16000       # Dışa açık hız
    _internal_sr = 48000      # RNNoise'ın beklediği hız

    def __init__(self):
        self.model = None

    def load(self):
        from pyrnnoise import RNNoise
        self.model = RNNoise(sample_rate=self._internal_sr)

    def process(self, audio: np.ndarray) -> np.ndarray:
        # 1) 16 kHz -> 48 kHz
        audio_48k = resample_poly(audio, up=self._internal_sr, down=self.sample_rate)

        # 2) float32 -> int16 ve (1, N) şekline getir (mono için tek kanal)
        audio_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)
        audio_2d = audio_int16[np.newaxis, :]   # (N,) -> (1, N)

        # 3) denoise_chunk generator; her frame için (speech_prob, denoised_frame) yield eder.
        #    Frame bölme, padding vb. işleri kütüphane kendisi hallediyor.
        frames = []
        for _, denoised_frame in self.model.denoise_chunk(audio_2d):
            frames.append(denoised_frame)

        # denoised_frame şekli (1, 480) -> birleştirip düzleştir
        denoised_int16 = np.concatenate(frames, axis=-1).squeeze(0)

        # 4) Orijinal 48 kHz uzunluğuna hizala (kütüphane son frame'i padleyebilir)
        denoised_int16 = denoised_int16[:len(audio_int16)]

        # 5) int16 -> float32
        denoised_float = denoised_int16.astype(np.float32) / 32767.0

        # 6) 48 kHz -> 16 kHz
        out = resample_poly(denoised_float, up=self.sample_rate, down=self._internal_sr).astype(np.float32)

        return out[:len(audio)]