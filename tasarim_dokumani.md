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

### 4.4 Pipeline Sıralaması — Akış 1: önce denoise → sonra konuşmacı seçimi

```
mic / .wav  →  Denoise  →  Speaker Selection  →  output
```

**Karar:** önce denoise, sonra konuşmacı seçimi (Akış 1).

**Gerekçe:**
- Temiz ses üzerinde VAD daha güvenilir karar verir (gürültülü seste false-positive yüksek).
- Mobil/telsiz senaryoda "yanlış konuşmacı seçimi" doğrudan fark edilen ürün hatasıdır.
- CPU israfı kabul edilebilir: denoiser tüm akışa uygulanır ama RTF < 0.1 yeterince düşük.

**Reddedilen alternatif (Akış 2):** Önce VAD, sadece konuşma pencerelerinde denoise.
Pencere sınırlarında bozulma + speaker selection adımının yine gürültüyle çalışması.

Modül arayüzü: `pipeline.base.BaseSpeakerSelector` ileride yazılacak — taslağı
[pipeline/README.md](pipeline/README.md) dosyasında. `BaseDenoiser` ile aynı
felsefe: `load()` + `process(audio, vad_segments=None)`.

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
├── benchmark/             # ölçüm ve raporlama altyapısı
│   ├── metrics.py         # süre, peak RSS, boyut, param, SI-SDR/STOI/PESQ
│   ├── mixer.py           # clean+noise -> hedef SNR karışımı
│   ├── runner.py          # tek model için load+process ölçümleri (warmup + N tekrar)
│   └── report.py          # CSV, XLSX, per-SNR pivot ve grafik üretimi
├── scripts/               # çalıştırılabilir benchmark giriş noktaları
│   ├── bench_real.py      # tek wav, sadece performans (RTF, peak RAM)
│   └── bench_synthetic.py # clean × noise × SNR, performans + kalite + grafikler
├── legacy/                # eski referans kod (artık kullanılmıyor)
│   └── benchmark_all.py
└── output/                # zaman damgalı deney klasörleri
    ├── bench_real_YYYYMMDD_HHMMSS/
    │   ├── 00_original.wav
    │   ├── NN_<model>.wav
    │   ├── results.csv
    │   └── results.xlsx
    └── bench_synthetic_YYYYMMDD_HHMMSS/
        ├── results_raw.csv
        ├── results_raw.xlsx
        ├── results_per_snr.xlsx
        └── plot_<metric>_vs_snr.png
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
- [x] İstatistiksel ölçüm (warmup + N tekrar, mean±std) — `benchmark/runner.py`
- [x] Peak RSS tabanlı RAM ölçümü — `PeakRSSTracker` (background thread, ~50ms örnekleme)
- [x] Objektif kalite metrikleri (PESQ, STOI, SI-SDR) — `benchmark/metrics.py`
- [x] Sentetik veri üretimi (temiz + gürültü, hedef SNR karışımı) — `benchmark/mixer.py`
- [x] İki ayrı benchmark scripti (gerçek wav vs sentetik) — `scripts/bench_real.py`, `scripts/bench_synthetic.py`
- [x] Per-SNR pivot raporları ve metrik-vs-SNR grafikleri — `benchmark/report.py`
- [x] Dataset indirme scripti (VCTK + Common Voice TR + DEMAND 7 sahne) — `scripts/download_data.py`
- [x] Profile sistemi (s_quick / s_smoke / m_medium) — `scripts/profiles.py`
- [x] Akıllı örnek seçici + `--save-strategy` (all / samples / none) — `benchmark/sampling.py`
- [x] Anomali yakalama (4 detektör: PESQ-RMS, varyans, trend, aşırı bastırma) — `benchmark/anomaly.py`
- [x] Otomatik hipotez testi (H1-H4) — `benchmark/hypothesis.py`
- [x] Tek-dosya HTML rapor (embedded audio + spektrogram + heatmap) — `benchmark/html_report.py`
- [x] Pipeline sıralaması kararı (Akış 1: önce denoise → sonra konuşmacı seçimi) — `pipeline/README.md`
- [x] Pre-built mix üretici (`build_mixes.py`) — `input_data/{profile}/` altında manifest + wav
- [x] `bench_synthetic` pre-built mix desteği (`--use-prebuilt auto/yes/no`)
- [x] Profil arası mix yeniden kullanımı (kopya optimizasyonu — manifest'te `source` sütunu)

### Sıradaki
- [ ] Baskın konuşmacı seçimi modülü (VAD + RMS) — **Pipeline sıralaması: Akış 1 (önce denoise)** kararlaştırıldı (Bölüm 4.4). Aday VAD'ler: Silero / WebRTC / saf enerji.
- [ ] **`m_medium` profili için veri seti genişletmek** — Profile 50 clean (25 EN + 25 TR) istiyor ama elimizde 10+10=20 var. Şu an `m_medium` = `s_quick` (script max_pairs'ı otomatik kırpıyor). Çözüm seçenekleri: `download_data.py` kotalarını yükseltmek, Türkçe için ek dataset eklemek, ya da profil tasarımını değiştirmek (n_repeats yükseltmek vb.). Detay: `development_task.md` Bölüm 5.
- [ ] Canlı mikrofon pipeline'ı (real-time test)
- [ ] Seçilen model(ler)in ONNX/TFLite dönüşümü
- [ ] Android tarafı için ölçüm ve entegrasyon

## 10. Kararlaştırılanlar
- **Baskın konuşmacı seçimi:** enerji tabanlı (VAD + RMS), ileride gerekirse diarization.
- **SNR seviyeleri:** -5, 0, 5, 10, 15 dB.
- **Ses formatı:** yakalama 48 kHz mono, işleme 16 kHz mono.
- **Giriş kaynakları:** canlı mikrofon + .wav dosyası + sentetik karışım.
- **Deney izolasyonu:** her çalıştırma zaman damgalı klasöre; eski sonuçlar silinmez.
