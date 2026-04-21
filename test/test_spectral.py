"""Basit dumanlı test: sentetik gürültülü sinyal üret, temizle, önce/sonra kaydet."""

import numpy as np
from audio_io.file_io import save_audio
from models.spectral_subtraction import SpectralSubtraction
from config import MODEL_SR


# 3 saniyelik 440 Hz "konuşma" (sinüs) + beyaz gürültü
duration = 3.0
t = np.linspace(0, duration, int(MODEL_SR * duration), endpoint=False)
clean = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
noise = 0.15 * np.random.randn(len(t)).astype(np.float32)

# İlk 100 ms tamamen sessiz -> modelin gürültü tahmin edebilmesi için
silence_samples = int(0.1 * MODEL_SR)
clean[:silence_samples] = 0.0

noisy = clean + noise

# Modeli yükle ve çalıştır
model = SpectralSubtraction()
model.load()
denoised = model.process(noisy)

# Önce/sonra kaydet, kulakla dinle
save_audio("noisy_input.wav", noisy)
save_audio("denoised_output.wav", denoised)

print(f"Gürültülü sinyal RMS : {np.sqrt(np.mean(noisy ** 2)):.4f}")
print(f"Temizlenmiş sinyal RMS: {np.sqrt(np.mean(denoised ** 2)):.4f}")
print("WAV'lar yazıldı: noisy_input.wav, denoised_output.wav")
