# NoiseSuppression — Denoiser Benchmark & Pipeline

Mikrofon girişini gürültüden temizleyen + baskın konuşmacıyı seçen sistemin
**model seçimi ve metodoloji** çalışması. Nihai hedef Android servis;
PC üzerindeki bu repo yedi denoiser modelini farklı senaryolarda karşılaştırır.

> **Tasarım dokümanı:** [`tasarim_dokumani.md`](tasarim_dokumani.md) (Türkçe)
> **Aktif planlama:** [`development_task.md`](development_task.md)

## Hızlı Başlangıç

### 1. Kurulum

```powershell
# Python 3.11 önerilir
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
# veya: source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

**Windows için ek bağımlılık:** PESQ paketi C derleyici gerektirir.
Visual Studio 2022 + "Desktop development with C++" workload kurulu olmalı.
Yoksa `pesq` import'u sessizce başarısız olur; PESQ metriği NaN döner ama
diğer metrikler etkilenmez.

### 2. Veri seti indirme

```powershell
python -m scripts.download_data
```

İndirdiği şeyler:
- **VCTK** — 10 İngilizce konuşma örneği (`data/clean/en/`)
- **Common Voice TR** — 10 Türkçe konuşma örneği (`data/clean/tr/`) ⚠ HuggingFace `huggingface-cli login` + dataset accept gerekir.
- **DEMAND** — 7 gürültü sahnesi (`data/noise/{TBUS,TCAR,TMETRO,SCAFE,SPSQUARE,OOFFICE,NPARK}/ch01.wav`)

Detay ve lisanslar: [`data/README.md`](data/README.md).

### 3. Mix üretimi (pre-built, opsiyonel ama önerilir)

```powershell
python -m scripts.build_mixes --profile s_smoke
```

`input_data/s_smoke/` altında **clean × noise × SNR kombinasyonlarının tümünü**
disk'e yazar (s_smoke için 56 mix). `manifest.csv` her dosyanın hangi clean +
hangi noise + hangi SNR'da olduğunu kaydeder, achieved SNR'u doğrular.

**Spot check (önerilir):** klasörü aç, 3-5 dosyayı rastgele dinle — gürültü
hedef SNR'da düzgün karışmış mı, kırpılma yok mu emin ol.

```powershell
explorer input_data\s_smoke
```

> Daha büyük profil için: `python -m scripts.build_mixes --profile s_quick`
> (20 clean × 9 noise × 5 SNR = 900 dosya, ~330 MB — 7 DEMAND sahnesi + sentetik PINK & WHITE)

### 4. Smoke test (5 dakikadan kısa)

```powershell
python -m scripts.bench_synthetic --profile s_smoke
```

Hızlı modellerle (spectral_subtraction, rnnoise, deepfilternet, metricgan_plus)
küçük bir benchmark yapar. Pre-built manifest (Adım 3 çıktısı) **zorunludur** —
yoksa "Önce: python -m scripts.build_mixes --profile s_smoke" diyerek çıkar.

### 5. Tam koşum — Senaryo S (kanonik)

```powershell
python -m scripts.bench_synthetic --profile s_quick
```

20 temiz × 7 gürültü × 5 SNR × 3 tekrar = 2100 ölçüm/model × 7 model.
CPU'da tahmini süre 30-60 dakika.

### 6. Raporu açma

```powershell
# Çıktı klasörünü bul
ls output/bench_synthetic_*/report.html

# Tarayıcıda aç
start output/bench_synthetic_20260512_165636/report.html
```

Rapor tek dosya HTML; e-mail'le paylaşılabilir, başka cihaza taşınınca da
çalışır (tüm audio + spektrogramlar base64 gömülü).

## VSCode Run Button

`.vscode/launch.json` hazır; **F5** veya Run panelinden konfigürasyon seç:

- **build_mixes (interactive profile)** → Pre-built mix üretimi, profil sorar
- **build_mixes (profile: s_smoke — fast)** → s_smoke mix'leri
- **build_mixes (profile: s_quick)** → s_quick mix'leri
- **bench_synthetic (interactive — F5 default)** → Profil + seçenek menüsü
- **bench_synthetic (profile: s_quick — full sweep)** → Senaryo S
- **bench_synthetic (profile: s_smoke — fast)** → Smoke test
- **bench_synthetic (profile: m_medium — large run)** → 50 pair, saatler
- **bench_real (auto-find wav, all models)** → Tek wav, sadece performans
- **bench_real (prompt for --wav)** → Wav yolunu sorar

Tüm konfigürasyonlar `.venv\Scripts\python.exe` interpreter'ını otomatik
seçer ve `PYTHONPATH` doğru ayarlanır.

## Mix doğrulama (spot check)

`build_mixes` çalıştıktan sonra `input_data/{profile}/` klasörünü manuel
incelemen önerilir. Dosya adlandırması doğrulamayı kolaylaştırır:

```
input_data/s_smoke/
├── manifest.csv                                  # tüm mix'lerin metadata'sı
├── en_spk5536_utt001__SCAFE__snr-05.wav         # SNR -5 (gürültü baskın)
├── en_spk5536_utt001__SCAFE__snr00.wav          # SNR 0 (eşit)
├── en_spk5536_utt001__SCAFE__snr05.wav          # SNR +5 (konuşma baskın)
├── tr_spkanon0003_utt004__TCAR__snr10.wav       # araç gürültüsü, +10 dB
└── ...
```

**Spot check listesi (3-5 dakika):**
1. `en_..._snr-05.wav` ile `en_..._snr10.wav` aynı clean'i kullanarak dinle.
   Gürültü seviyesi sayısal SNR'a uyuyor mu?
2. Farklı sahnelerden (SCAFE vs TCAR vs NPARK) örnek dinle. Doğru gürültü
   içeriği geldi mi?
3. Hem `en_*` hem `tr_*` örneği dinle. Konuşmacı kimliği belli mi?
4. `manifest.csv`'ye bak: `achieved_snr_db` ile `target_snr_db` arası fark
   < 0.5 dB olmalı.

Eğer dinleme sonunda mix'lerden tatmin olmadıysan parametre ayarı için
[`scripts/profiles.py`](scripts/profiles.py) veya yeniden planlama için
[`development_task.md`](development_task.md) güncellenir.

## Çıktı klasör yapısı (bench_synthetic)

```
output/bench_synthetic_YYYYMMDD_HHMMSS/
├── samples/                          # --save-strategy=samples (varsayılan)
│   ├── _reference/
│   │   ├── clean_pair00_en.wav
│   │   └── noisy_{SCENE}_snr{N}dB.wav
│   └── {model_name}/
│       └── {SCENE}_snr{N}dB.wav
├── results_raw.csv                   # tüm ölçüm satırları
├── results_raw.xlsx                  # formatlı versiyon
├── results_per_snr.xlsx              # 5 sheet (PESQ/SI-SDR/STOI/RTF/peak_RAM)
├── anomalies.csv                     # yakalanan anomaliler (varsa)
├── plot_*.png                        # metric-vs-SNR grafikleri
└── report.html                       # 8 bölümlü tek-dosya rapor (~6-25 MB)
```

## CLI argümanları (özet)

```
python -m scripts.bench_synthetic [--profile NAME] [overrides...]

--profile {s_quick,s_smoke,m_medium}   önceden tanımlı parametre seti
--clean-dir DIR                        temiz konuşma kaynağı (varsayılan: data/clean)
--noise-dir DIR                        gürültü kaynağı (varsayılan: data/noise)
--snrs FLOAT [FLOAT ...]               SNR seviyeleri (dB)
--max-pairs N                          rastgele (clean, noise) çifti sayısı
--n-repeats N                          her ölçüm için process() tekrar sayısı
--models LIST                          "all" veya "rnnoise,deepfilternet,..."
--save-strategy {all,samples,none}     wav kayıt stratejisi (varsayılan: samples)
--no-html-report                       HTML raporu üretme
--no-warmup                            warmup process() atla
--seed N                               örnekleme tohumu (varsayılan: 42)
```

Tek wav üzerinde sadece performans için:

```
python -m scripts.bench_real --wav input.wav [--models all]
```

`--wav` verilmezse `test/` ve `data/clean/` klasörlerinden ilk bulduğu wav'ı kullanır.

## Klasör organizasyonu

```
NoiseSuppression/
├── config.py                       # MODEL_SR=16000, vb. proje sabitleri
├── audio_io/file_io.py             # load_audio (16 kHz mono normalize), save_audio
├── models/                         # 7 denoiser wrapper + BaseDenoiser
├── benchmark/                      # ölçüm + raporlama
│   ├── metrics.py                  # zaman, peak RSS, PESQ/STOI/SI-SDR, RMS, HF ratio
│   ├── mixer.py                    # mix_at_snr (RMS-scaled)
│   ├── runner.py                   # warmup + N-repeat process ölçümü
│   ├── sampling.py                 # 28 dinleme örneği seçici
│   ├── anomaly.py                  # 4 anomali detektörü
│   ├── spectrogram.py              # log-mel spektrogram PNG'leri
│   ├── hypothesis.py               # H1-H4 otomatik test
│   ├── html_report.py              # 8 bölümlü tek-dosya HTML rapor
│   └── report.py                   # CSV, XLSX, per-SNR pivot, matplotlib
├── scripts/                        # CLI giriş noktaları
│   ├── bench_real.py
│   ├── bench_synthetic.py
│   ├── download_data.py
│   ├── profiles.py                 # s_quick/s_smoke/m_medium
│   └── _model_registry.py
├── pipeline/                       # speaker selection (placeholder, ileride)
├── legacy/                         # eski benchmark_all.py (referans)
├── test/                           # birim testler (test_anomaly.py vb.)
├── data/                           # ses verisi (gitignored, indir)
├── output/                         # zaman damgalı sonuçlar (gitignored)
├── pretrained_models/              # SpeechBrain cache (gitignored)
├── .venv/                          # Python venv (gitignored)
├── requirements.txt
├── tasarim_dokumani.md             # ana tasarım dokümanı
├── development_task.md             # aktif planlama
└── README.md                       # bu dosya
```

## Modeller

| Model | RTF (10 s wav, CPU) | Peak RAM | Params | Açıklama |
|---|---|---|---|---|
| spectral_subtraction | 0.004 | 230 MB | 0 | Klasik DSP baseline, ML yok |
| metricgan_plus | 0.020 | 1 GB | 1.9 M | GAN, PESQ-optimize, agresif olabilir |
| deepfilternet | 0.046 | 420 MB | 2.1 M | **Mobil için en güçlü aday** |
| demucs_dns48 | 0.050 | 740 MB | 18.9 M | |
| rnnoise | 0.069 | 270 MB | 0 | C kütüphanesi, hafif |
| demucs_dns64 | 0.081 | 930 MB | 33.5 M | |
| demucs_master64 | 0.081 | 1 GB | 33.5 M | Tüm Demucs varyantları arasında en büyük |

Detay sonuçlar: `output/bench_synthetic_*/report.html` raporlarında.

## Lisans & atıflar

Kod henüz lisanslanmamış (özel araştırma projesi). Üçüncü taraf datasetler ve
modeller kendi lisanslarına tabidir — detay [`data/README.md`](data/README.md).
