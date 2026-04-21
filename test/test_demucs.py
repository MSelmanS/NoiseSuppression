"""Demucs/Denoiser test. Kullanıcıdan dosya seçtirir ya da sentetik sinyal üretir."""

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from audio_io.file_io import load_audio, save_audio
from models.demucs_model import DemucsDenoiser
from config import MODEL_SR


def ask_user_for_file():
    root = tk.Tk()
    root.withdraw()
    want_file = messagebox.askyesno(
        title="Giriş seçimi",
        message="Kendi .wav dosyanı seçmek ister misin?\n(Hayır seçersen sentetik sinyal kullanılır)"
    )
    path = None
    if want_file:
        path = filedialog.askopenfilename(
            title="Gürültülü .wav dosyası seç",
            filetypes=[("WAV dosyaları", "*.wav"), ("Tüm dosyalar", "*.*")]
        )
        if not path:
            path = None
    root.destroy()
    return path


def make_synthetic():
    duration = 3.0
    t = np.linspace(0, duration, int(MODEL_SR * duration), endpoint=False)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = 0.15 * np.random.randn(len(t)).astype(np.float32)
    clean[:int(0.1 * MODEL_SR)] = 0.0
    return clean + noise


# --- Ana akış ---
path = ask_user_for_file()

if path is None:
    print("Sentetik sinyal kullanılacak.")
    noisy = make_synthetic()
else:
    print(f"Seçilen dosya: {path}")
    noisy, _ = load_audio(path)

print("Model yükleniyor (ilk çağrıda ağırlıklar indirilir)...")
model = DemucsDenoiser()
model.load()

print("Temizleniyor...")
denoised = model.process(noisy)

save_audio("noisy_input_demucs.wav", noisy)
save_audio("denoised_demucs.wav", denoised)

print(f"Gürültülü RMS : {np.sqrt(np.mean(noisy ** 2)):.4f}")
print(f"Temizlenmiş RMS: {np.sqrt(np.mean(denoised ** 2)):.4f}")
print("WAV'lar yazıldı: noisy_input_demucs.wav, denoised_demucs.wav")
