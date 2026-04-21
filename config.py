"""Proje geneli sabitler. Tek noktadan yönetim için her yerde buradan import edilir."""

# --- Ses formatı ---
CAPTURE_SR = 48000   # Mikrofondan yakalama hızı (PC donanımı doğal olarak 48 kHz verir)
MODEL_SR   = 16000   # Modellerin çalıştığı hız (RNNoise/DeepFilterNet vb. 16 kHz'de eğitildi)
CHANNELS   = 1       # Mono ile çalışıyoruz

# --- Klasörler ---
DATA_DIR    = "data"      # ham ve işlenmiş ses verileri
RESULTS_DIR = "results"   # çıktı sesleri ve rapor dosyaları
