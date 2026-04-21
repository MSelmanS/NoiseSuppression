# Ses İzolasyonu ve Konuşmacı Seçimi — Tasarım Dokümanı

> Bu doküman sistemin **tasarımını** anlatır: amaç, kapsam, problem tanımı, mimari, veri stratejisi ve yol haritası. Model karşılaştırması ve deney sonuçları **ayrı bir sonuç raporunda** tutulur.

---

## 1. Amaç
Mikrofondan alınan ses akışını gürültüden temizlemek ve mikrofona en yakın (baskın) konuşmacının sesini çıkışa vermek. Nihai hedef: **Android üzerinde çalışan bir servis**. PC üzerindeki bu çalışma model seçimi ve metodoloji araştırmasıdır.

## 2. Kapsam
**Dahil:** Tek kanallı ses girişi, gürültü bastırma, baskın konuşmacı seçimi, çoklu-model karşılaştırma altyapısı, PC üzerinde prototip.

**Hariç (şimdilik):** Çok mikrofonlu beamforming, konuşmacı kimliklendirme, Android uygulamasının kendisi (model seçimi netleştikten sonra ayrı iş olarak ele alınacak).

## 3. Problemin İki Alt Parçası

### 3.1 Gürültü Bastırma (Speech Enhancement)
Konuşma dışı sesleri (fan, trafik, klavye, uğultu vb.) bastırarak konuşmayı öne çıkarma. Literatürde olgun bir alan; hem klasik DSP hem derin öğrenme çözümleri mevcut.

### 3.2 Baskın / En Yakın Konuşmacı Seçimi
Tek mikrofonla fiziksel mesafe doğrudan ölçülemez. **Enerji tabanlı** yaklaşım kullanılacak:
- VAD (Voice Activity Detection) ile konuşma bölgelerini bul.
- Kısa pencerelerde RMS/enerji hesapla, baskın (yüksek enerjili) konuşmayı koru, zayıf arka plan konuşmalarını bastır.
- Diarization gibi gelişmiş yöntemler ileriki aşamada gerekirse eklenir.

## 4. Mimari Prensipler

### 4.1 Ortak Model Arayüzü
Her denoiser modeli `BaseDenoiser` soyut sınıfından türer ve aynı iki metodu sunar:
- `load()` — ağırlıkları/kaynakları yükle
- `process(audio) -> audio` — gürültülü sesi temizle

Bu sayede benchmark ve pipeline kodu modelden bağımsız çalışır; yeni model eklemek tek bir wrapper dosyası yazmak demektir.

### 4.2 Ses Formatı Standardı
- **Yakalama:** 48 kHz mono, 16-bit PCM (PC mikrofonu doğal formatı).
- **Model işleme:** 16 kHz mono — aday modellerin çoğu bu frekansta eğitildi, mobil hedefle de uyumlu.
- **Çıkış:** 16 kHz (gerekirse 48 kHz'e yükseltilir).

Farklı örnekleme hızında çalışan modellerin (RNNoise 48 kHz, DeepFilterNet içsel 48 kHz) resample mantığı wrapper içinde gizlenir; dışarıdan bakınca hepsi 16 kHz çalışır.

### 4.3 Giriş Kaynakları
Sistem üç tür girişi destekler; pipeline aynı, sadece ses kaynağı değişir:

1. **Kayıtlı .wav dosyası** (kullanıcı dosya seçer)
2. **Canlı mikrofon** (real-time test için — ilerleyen aşama)
3. **Sentetik karışım** (temiz + gürültü, objektif kalite ölçümü için)

## 5. Karşılaştırma Metrikleri

Modeller iki eksende değerlendirilir:

### 5.1 Performans (Mobil Uygunluk)
- **RTF (Real Time Factor)** = işlem süresi / ses süresi. <1 ise real-time mümkün.
- **Çıkarım süresi** (mutlak, saniye cinsinden)
- **Model yükleme süresi**
- **RAM kullanımı** (yükleme + işlem sırasında)
- **Model boyutu** (MB) ve **parametre sayısı** — APK boyutu ve mobil CPU uyumu
- **ONNX/TFLite'a dönüşüm uygunluğu** (ileriki aşama)

### 5.2 Kalite
- **Kulak testi** — erken aşamada sübjektif izlenim
- **PESQ, STOI, SI-SDR** — objektif metrikler, yalnızca temiz referansın bilindiği sentetik veride hesaplanabilir (ileriki aşama)

## 6. Veri Stratejisi

### 6.1 Gerçek Dünya Testi
- PC mikrofonu veya kayıtlı .wav dosyası.
- **Amaç:** performans ölçümü (RTF, RAM) ve kulakla kalite kontrolü.
- **Sınırı:** temiz referans olmadığı için objektif kalite metrikleri hesaplanamaz.

### 6.2 Sentetik Karışım (ileriki aşama)
- **Temiz konuşma:** LibriSpeech veya VCTK.
- **Gürültü:** DEMAND veya MUSAN.
- **SNR seviyeleri:** -5, 0, 5, 10, 15 dB (DNS Challenge ve VoiceBank-DEMAND benchmark'larıyla uyumlu standart aralık).
- **Amaç:** PESQ/STOI/SI-SDR ile objektif karşılaştırma.

## 7. Proje Yapısı
```
project/
├── config.py              # proje sabitleri (SR, yollar)
├── audio_io/              # ses dosyası okuma/yazma
│   └── file_io.py
├── models/                # denoiser wrapper'ları
│   ├── base.py            # ortak arayüz (BaseDenoiser)
│   ├── spectral_subtraction.py
│   ├── rnnoise_model.py
│   ├── deepfilternet_model.py
│   ├── demucs_model.py    # dns48/dns64/master64 varyantları
│   └── metricgan_model.py
├── benchmark/             # ölçüm yardımcıları
│   └── metrics.py         # süre, RAM, boyut, parametre sayısı
├── benchmark_all.py       # toplu karşılaştırma scripti
└── output/                # zaman damgalı deney klasörleri
    └── output_YYYYMMDD_HHMMSS/
        ├── 00_original.wav
        ├── NN_<model>.wav
        ├── results.csv
        └── results.xlsx
```

## 8. Deney Çıktı Stratejisi
Her benchmark çalıştırması kendi zaman damgalı klasörüne yazar — önceki deneyler **silinmez**, karşılaştırma ve yeniden denetim mümkün olur. Çıktılar:
- Her modelin temizlenmiş `.wav`'ı (kulak testi için)
- Ham orijinal `.wav` (referans)
- `results.csv` (ham veri arşivi)
- `results.xlsx` (formatlı rapor)

## 9. Yol Haritası

### Tamamlanan
- [x] Proje iskeleti + ortak model arayüzü (`BaseDenoiser`)
- [x] Ses I/O modülü (okuma, yazma, resample)
- [x] Model wrapper'ları: Spectral Subtraction, RNNoise, DeepFilterNet, Demucs (3 varyant), MetricGAN+
- [x] Benchmark altyapısı (RTF, süre, RAM, boyut, parametre sayısı)
- [x] Sonuç raporlama (CSV + formatlı XLSX)

### Sıradaki
- [ ] Objektif kalite metrikleri (PESQ, STOI, SI-SDR)
- [ ] Sentetik veri üretimi (temiz + gürültü, farklı SNR seviyeleri)
- [ ] Baskın konuşmacı seçimi modülü (VAD + RMS)
- [ ] Canlı mikrofon pipeline'ı (real-time test)
- [ ] Seçilen model(ler)in ONNX/TFLite dönüşümü
- [ ] Android tarafı için ölçüm ve entegrasyon

## 10. Kararlaştırılanlar
- **Baskın konuşmacı seçimi:** enerji tabanlı (VAD + RMS), ileride gerekirse diarization.
- **SNR seviyeleri:** -5, 0, 5, 10, 15 dB.
- **Ses formatı:** yakalama 48 kHz mono, işleme 16 kHz mono.
- **Giriş kaynakları:** canlı mikrofon + .wav dosyası + sentetik karışım.
- **Deney izolasyonu:** her çalıştırma zaman damgalı klasöre; eski sonuçlar silinmez.
