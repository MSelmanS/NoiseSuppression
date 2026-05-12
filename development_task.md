# Development Task — Handoff Document

> **Purpose.** Bu dosya iki Claude arasında bir köprü:
> - **Claude (chat / claude.ai)** planlamayı yapar, bu dosyayı yazar.
> - **Claude Code (IDE)** dosyayı okur, taskları sırayla uygular.
>
> Bu turun odağı: **kapsamlı çok-değişkenli denoiser benchmark altyapısının veri ve raporlama yönleriyle tamamlanması.** Konuşmacı seçimi modülü bu turun konusu DEĞİL — pipeline'da yeri ayrılır, kendisi sonra eklenir.

---

## 1. Bu turun bağlamı

### 1.1 Mevcut durum (kısa özet)
- Benchmark altyapısı çalışıyor: `BaseDenoiser` arayüzü, 7 model wrapper, `bench_synthetic.py` clean × noise × SNR sweep yapıyor, `n_repeats` ile mean±std hesaplıyor, PESQ/STOI/SI-SDR/RTF/peak RAM ölçülüyor, CSV + XLSX + per-SNR pivot + matplotlib grafikleri üretiliyor.
- Eksik olan: **gerçek veri seti** (`data/clean/`, `data/noise/` boş veya yer tutucu) ve **kapsamlı rapor formatı** (mevcut çıktı dağınık dosyalar halinde, dinleme/değerlendirme zor).

### 1.2 Bu turun araştırma sorusu
"Yedi denoiser modelinin **farklı gürültü tiplerinde, farklı SNR seviyelerinde, hem Türkçe hem İngilizce konuşmada** karşılaştırmalı performansı nedir; hangi(leri) mobil/telsiz hedefimiz için en uygun adaydır?"

### 1.3 Kullanıcı senaryosu (veri ve test profilinin dayanağı)
- **Birincil hedef:** Türkçe konuşan kullanıcı, telsiz benzeri cihaz
- **Sahne ağırlığı:** %80 Türkiye — karışık ama trafik/dış mekan ağırlıklı (motor, sokak, rüzgar) + ofis/kafe/karışık ortamlar
- **İkincil hedef:** Çok dilli destek için İngilizce de test edilir (literatür kıyaslama + dağıtım dışı performans)

### 1.4 Önceden kayıt altına alınan hipotez
Bu hipotez deney **sonuçları görülmeden önce** kayıt altına alındı; rapor onayı/çürütmesini otomatik test edecek.

> Klasik/hafif yöntemler (spectral_subtraction, RNNoise) düşük SNR ve konuşmacı gürültüsü gibi karmaşık ortamlarda zorlanır. DL modelleri (DeepFilterNet, Demucs varyantları) fan/motor/uğultu gibi sürekli gürültülerde daha iyi sonuç verir. MetricGAN+ bazı düşük SNR senaryolarında gürültüyü agresif bastırırken konuşma doğallığını bozar. Kafe/kalabalık konuşma gibi hedef konuşmaya benzeyen gürültülerde tüm modeller zorlanır; model farkları en net burada ortaya çıkar.

Hipotez 4 alt-iddiaya ayrıştırılır (otomatik test için):

- **H1.** SNR ≤ 0 dB'de spectral_subtraction + rnnoise'ın ortalama PESQ'si, DL modellerinin (DFN, Demucs varyantları, MetricGAN+) ortalama PESQ'sinden ≥ 0.3 düşüktür.
- **H2.** Sürekli/sabit gürültü sahnelerinde (DKITCHEN, OOFFICE, TCAR, NPARK) DFN ve Demucs ailesinin ortalama PESQ'si, klasik yöntemlerden ≥ 0.4 yüksektir.
- **H3.** MetricGAN+ çıkışlarında, en az iki SNR seviyesinde ortalama RMS düşüşü > 8 dB veya HF Energy Ratio < 0.0010 (aşırı agresiflik göstergesi).
- **H4.** Kafe/kalabalık konuşma sahnesinde (SCAFE, SPSQUARE) tüm modellerin PESQ ortalamaları yakın (max−min < 0.4) — model farkı azalır, ortak zorluk vardır.

---

## 2. Karar özeti (binding)

| Konu | Karar |
|---|---|
| **Veri stratejisi** | Strateji B: VCTK alt küme + Common Voice TR + DEMAND 7 sahne |
| **Senaryo** | S: 20 temiz × 7 gürültü × 5 SNR × 3 tekrar = 2100 işlem/model |
| **SNR seviyeleri** | −5, 0, 5, 10, 15 dB (zaten karar) |
| **DEMAND sahneleri** | TBUS, TCAR, TMETRO (trafik/araç), SCAFE, SPSQUARE (kalabalık konuşma), OOFFICE (sabit gürültü), NPARK (rüzgar/doğa) |
| **Temiz konuşma** | 10 İngilizce klip (VCTK) + 10 Türkçe klip (CV-TR) — 50/50 dengeli |
| **Rapor formatı** | HTML tek dosya, embedded audio + spektrogram + otomatik hipotez testi |
| **Dinleme galerisi** | 2 kritik sahne (SCAFE, TCAR) × 2 SNR (−5, 5) × 7 model = 28 örnek |
| **Anomali yakalama** | Metrik-RMS çelişkisi, yüksek std, beklenmedik düşüş |
| **Konuşmacı seçimi** | Bu turda EKLENMEYECEK; pipeline mimarisinde yeri ayrılır |
| **Pipeline akışı** | Akış 1 → önce denoise → sonra konuşmacı seçimi (sonraki tur) |
| **Hipotez kaydı** | Bu dokümanın 1.4'ünde sabit; rapor otomatik onaylar/çürütür |

---

## 3. Numaralı Task Listesi

Tasklar **sırayla** uygulanacak. Her task acceptance criteria'ları geçince commit edilir.

---

### Task 1 — Dataset indirme ve hazırlama scripti

- **Why:** `data/clean/` ve `data/noise/` şu an yer tutucu; gerçek benchmark için VCTK alt küme + Common Voice TR + DEMAND sahneleri lazım. Bu adım olmadan bench_synthetic anlamlı çalışmaz.
- **Files to touch:**
  - `scripts/download_data.py` (new)
  - `data/README.md` (new) — dataset kaynakları, lisansları, dosya organizasyonu
  - `requirements.txt` (eğer ek paket gerekirse — `datasets`, `librosa` vs.)
- **Approach:**
  - **VCTK:** `https://datashare.ed.ac.uk/handle/10283/2950` üzerinden veya HuggingFace `datasets` üzerinden. **Alt küme:** 10 farklı konuşmacı (5 erkek + 5 kadın, çeşitli aksanlar), her birinden 1 cümle — toplam 10 dosya. Hedef yol: `data/clean/en/spk{ID}_uttN.wav`
  - **Common Voice Turkish:** HuggingFace `mozilla-foundation/common_voice_*_0` üzerinden TR alt küme. **Alt küme:** kalite filtresi (up_votes ≥ 2, down_votes == 0) sonrasından 10 dosya, çeşitli konuşmacılar. Hedef yol: `data/clean/tr/spkN_uttN.wav`
  - **DEMAND:** `https://zenodo.org/records/1227121` üzerinden. **Sadece 7 sahne** indir: TBUS, TCAR, TMETRO, SCAFE, SPSQUARE, OOFFICE, NPARK. Her sahnenin ch01.wav'ı yeterli (1 mikrofon kanalı, ~5 dk). Hedef yol: `data/noise/{SCENE}/ch01.wav`
  - Tüm sesleri **16 kHz mono float32**'ye normalize ederek kaydet (`audio_io.file_io.load_audio` ile yükle, `save_audio` ile yaz).
  - Script idempotent: ikinci çağrıda mevcut dosyaları yeniden indirmez.
  - İndirme hatalarında nazikçe başarısız ol, hangi dosyanın inmediğini logla, diğerlerine devam et.
- **Acceptance criteria:**
  - [ ] `python -m scripts.download_data` çalıştırınca `data/clean/en/` altında 10, `data/clean/tr/` altında 10 wav dosyası oluşur
  - [ ] `data/noise/` altında 7 alt klasör (sahne adıyla) ve her birinde `ch01.wav` oluşur
  - [ ] Tüm wav dosyaları 16 kHz mono (kontrol: `soundfile.info`)
  - [ ] `data/README.md` her dataset için kaynak URL, lisans, dosya sayısı ve organizasyonu açıklar
  - [ ] Script ikinci çalıştırmada "zaten var, atlanıyor" mesajıyla hızlıca çıkar
- **Out of scope:** Tüm VCTK / CV / MUSAN'ı indirmek; başka sahne eklemek; veri augmentasyonu
- **Depends on:** None

---

### Task 2 — Bench_synthetic için varsayılan dataset profili

- **Why:** Mevcut `bench_synthetic.py` CLI argümanlarıyla çalışıyor. Senaryo S için her seferinde uzun argüman yazmak yerine, bu çalışmanın "kanonik konfigürasyonu" tek yerde tanımlı olmalı.
- **Files to touch:**
  - `scripts/bench_synthetic.py` → `--profile` argümanı ekle
  - `scripts/profiles.py` (new) — profile sözlükleri
  - `.vscode/launch.json` — yeni profile çağrısı için config
- **Approach:**
  - `profiles.py` içine `PROFILES` sözlüğü: her profile bir dict (clean_dir, noise_dir, snrs, max_pairs, n_repeats, models).
  - Tanımlanacak profiller:
    - **`s_quick`** — Senaryo S (20 temiz × 7 noise × 5 SNR × 3 rep, all models)
    - **`s_smoke`** — debug için ufak (4 temiz × 2 noise × 2 SNR × 1 rep, fast models only)
    - **`m_medium`** — Senaryo M (50 × 7 × 5 × 3) — gelecekte gerekirse
  - `bench_synthetic.py`'ye `--profile NAME` argümanı eklenir. Verilirse profile dict'inden değerleri okur; ayrı CLI argümanları override edebilir.
  - Profile seçilince konsola net log: "Profile=s_quick, ~2100 işlem/model, tahmini süre 30-60dk".
- **Acceptance criteria:**
  - [ ] `python -m scripts.bench_synthetic --profile s_smoke` 5 dakikadan kısa sürede tamamlanır
  - [ ] `python -m scripts.bench_synthetic --profile s_quick` ile S senaryosu çalışır
  - [ ] Profil overrides çalışır: `--profile s_quick --max-pairs 10` 20 yerine 10 pair kullanır
  - [ ] Profile bilinmiyorsa anlamlı hata mesajı
- **Out of scope:** Yeni model eklemek; metrik değiştirmek
- **Depends on:** Task 1

---

### Task 3 — Akıllı örnek seçici (28 dinleme örneği için)

- **Why:** Senaryo S çalıştığında 14.700 wav dosyası üretilebilir. Hepsini diskte tutmak gereksiz. Bunun yerine sadece **rapor için seçilmiş 28 örneği** kaydedip kalanları (hatta hiç dosyaya yazmadan) çöpe atalım.
- **Files to touch:**
  - `benchmark/sampling.py` (new) — örnek seçici mantığı
  - `scripts/bench_synthetic.py` — kaydetme kararı (hangi örnekleri tutar)
- **Approach:**
  - Yeni argüman `--save-strategy {all, samples, none}`:
    - `all` — mevcut davranış, hepsini kaydet (gereksiz disk yükü)
    - `samples` — sadece dinleme örnekleri için seçilen kombinasyonları kaydet (varsayılan)
    - `none` — hiç kaydetme, sadece metrikleri tut
  - `sampling.py` içinde fonksiyon: `pick_listening_samples(scenes, snrs, models)`. Hangi (sahne, snr, model, pair) kombinasyonlarının saklanacağını döndürür:
    - Kritik sahneler: **SCAFE** ve **TCAR** (hipotezdeki en bilgilendirici sahneler)
    - Kritik SNR'lar: **−5 dB** ve **5 dB** (en zor + orta zorluk)
    - Her kombinasyondan 1 pair seçilir (deterministik, seed=42)
    - Bu da 2 × 2 × 7 = 28 ses → 28 dosya
  - Ek olarak her **gürültülü mix** ve **temiz referans** dosyası da bir kez kaydedilir (model bağımsız). Yani input dosyaları: 2 sahne × 2 SNR × 1 pair = 4 noisy mix + 1 clean reference = **toplam 5 input dosyası**.
  - Çıktı klasör yapısı:
    ```
    output/bench_synthetic_YYYYMMDD_HHMMSS/
    ├── samples/
    │   ├── _reference/
    │   │   ├── clean_pair0_en.wav
    │   │   ├── noisy_SCAFE_snr-5dB.wav
    │   │   └── ... (4 noisy mix)
    │   └── {model_name}/
    │       └── SCAFE_snr-5dB.wav   (her modelden 4 çıktı)
    ├── results_raw.csv
    ├── results_raw.xlsx
    ├── results_per_snr.xlsx
    └── plot_*.png
    ```
- **Acceptance criteria:**
  - [ ] `--save-strategy samples` ile `output/.../samples/` altında en fazla 28 + 5 dosya bulunur (toplam ≤ 33 wav)
  - [ ] `--save-strategy none` ile hiç wav kaydedilmez, sadece raporlar
  - [ ] `--save-strategy all` mevcut davranışla aynı (geriye uyumluluk)
  - [ ] Aynı seed=42 ile çalıştırma seçilen pair'ları tekrarlanabilir biçimde verir
- **Out of scope:** Spektrogram üretimi (Task 5'te)
- **Depends on:** Task 2

---

### Task 4 — Anomali yakalama mantığı

- **Why:** Sayısal metrikler yanıltabilir (MetricGAN+ örneği → PESQ iyi ama ses kötü). Otomatik bir "şüpheli sonuç" listesi rapora ek değer katar.
- **Files to touch:**
  - `benchmark/anomaly.py` (new) — anomali detektörleri
  - `benchmark/report.py` — anomali listesini rapora gömme
- **Approach:**
  - Şu kuralları uygula (sonuç dataframe üzerinde):
    1. **PESQ-RMS çelişkisi:** PESQ ≥ mean(PESQ) + 0.3 ama o satırın temiz çıktısının RMS'i orijinal noisy'nin RMS'inden 6 dB+ düşük → "Skor iyi ama ses kırpılmış" anomalisi
    2. **Yüksek varyans:** Aynı (model, scene, snr) kombinasyonunda PESQ std > 0.5 → "Tutarsız davranış" anomalisi
    3. **Beklenen düzenden sapma:** SNR arttıkça PESQ artmalı; ardışık iki SNR'da PESQ düşerse → "Anormal trend" anomalisi
    4. **MetricGAN-tipi aşırı bastırma:** HF Ratio < 0.0010 veya çıktı RMS < orijinal − 10 dB → "Aşırı agresif" anomalisi
  - Her anomalili satır için: (model, scene, snr, pair, anomaly_type, severity, related_metrics) içeren bir kayıt üret.
  - HTML raporunda ayrı bir "Anomaliler" bölümünde göster (Task 5'te birleşecek).
- **Acceptance criteria:**
  - [ ] `detect_anomalies(results_df)` çağrısı bir DataFrame döner; her satır anomali türü ve ilgili satıra referans içerir
  - [ ] Senaryo S sonuçları üzerinde çalıştırıldığında en az 1, en fazla 50 anomali yakalanır (eşik kalibrasyonu)
  - [ ] Birim test: yapay PESQ-RMS çelişkisi içeren mock DataFrame'de anomali doğru yakalanır
- **Out of scope:** Anomaliyi otomatik düzeltme; ek metrik ekleme
- **Depends on:** Task 2

---

### Task 5 — HTML rapor üreticisi

- **Why:** Mevcut çıktı (CSV + XLSX + ayrı PNG'ler) dağınık. Tek dosyada tüm metrikleri + embedded audio + spektrogram + hipotez testi + anomalileri sunan HTML rapor lazım.
- **Files to touch:**
  - `benchmark/html_report.py` (new) — ana üretici
  - `benchmark/spectrogram.py` (new) — spektrogram PNG'lerini üreten yardımcı
  - `benchmark/hypothesis.py` (new) — Bölüm 1.4'teki H1-H4 otomatik test
  - `scripts/bench_synthetic.py` — bench sonunda html_report çağrısı
- **Approach:**
  - **Tek HTML dosya**, audio dosyaları ve PNG'ler **base64 embedded** (rapor tek dosya olarak paylaşılabilsin). Toplam dosya boyutu hedefi: < 25 MB.
  - **Şablon:** Jinja2 yerine f-string + heredoc; basit tutalım, harici şablon dosyası yok.
  - **Bölümler (sırayla):**
    1. **Başlık & meta** — deney tarihi, profile adı, dataset, model listesi, hipotez metni (1.4'ten)
    2. **Liderlik tablosu** — her metrik için (PESQ, STOI, SI-SDR, RTF, peak_RAM) model sıralaması ve ortalamalar
    3. **Hipotez testi** — H1-H4 için tek tek "DOĞRULANDI / KISMEN / ÇÜRÜTÜLDÜ" + dayanak rakamlar (`benchmark.hypothesis` üretir)
    4. **Sahne × SNR ısı haritaları** — her model × metrik için, color-coded HTML tabloları
    5. **Detay tabloları** — model × sahne × SNR ortalama ± std (mevcut per-SNR XLSX'in HTML versiyonu)
    6. **Dinleme galerisi** — 28 örnek, her biri için:
       - Sol: orijinal gürültülü `<audio controls>` + spektrogram PNG
       - Sağ: temizlenmiş `<audio controls>` + spektrogram PNG
       - Altta: o örneğin PESQ/STOI/SI-SDR değerleri
    7. **Anomaliler** — Task 4 çıktısı tablo halinde
    8. **Grafikler** — PESQ vs SNR (model-cluster), RTF vs model bar chart, peak_RAM bar chart (mevcut PNG'ler embedded)
  - Spektrogram: `librosa.feature.melspectrogram` + `librosa.display.specshow` ile log-mel spektrogram (256 mel bins, hop 256). Renk skalası tüm spektrogramlarda aynı olsun (vmin/vmax dataset minmax) — modeller arası gözle kıyas yapılabilsin.
  - Hipotez testi otomatik kurallı: Bölüm 1.4'teki H1-H4 koşulları kod halinde; her biri için sonuç dict'i {iddia: H1, status: VERIFIED/PARTIAL/REJECTED, evidence: {...}}.
- **Acceptance criteria:**
  - [ ] Senaryo S sonunda `output/.../report.html` üretilir, < 25 MB
  - [ ] Tarayıcıda açıldığında 8 bölüm de görünür ve gezinilebilir
  - [ ] 28 ses örneği play butonuyla çalınabilir
  - [ ] 28 örnek için spektrogram (orijinal + temiz) yan yana görünür
  - [ ] Hipotez testi her 4 alt-iddiayı sayısal kanıtla işaretler
  - [ ] Rapor başka bir cihaza taşınınca (sadece HTML, ek dosya yok) bozulmadan açılır
- **Out of scope:** İnteraktif grafikler (plotly/bokeh — bu turda statik PNG yeterli); LLM-destekli yorumlar (gelecek hedef)
- **Depends on:** Task 3, Task 4

---

### Task 6 — Pipeline mimari kararını dokümante et

- **Why:** Konuşmacı seçimi bu turda yazılmıyor ama gelecek için mimari karar şimdi kayıtlı olmalı. Aksi halde sonraki turda "denoise mi önce, VAD mı önce" tartışması tekrarlanır.
- **Files to touch:**
  - `tasarim_dokumani.md` — Bölüm 4'e ek "Pipeline Sıralaması" alt-bölümü
  - `pipeline/__init__.py` (new) — boş paket, placeholder
  - `pipeline/README.md` (new) — Akış 1 prensibinin kısa açıklaması, BaseDenoiser → BaseSpeakerSelector arayüz iskeleti taslağı (yorum olarak)
- **Approach:**
  - Tasarım dokümanı 4.4 olarak: "Pipeline Sıralaması — Akış 1: önce denoise → sonra konuşmacı seçimi. Gerekçe: temiz ses üzerinde VAD daha güvenilir karar verir; mobil senaryoda 'yanlış konuşmacı seçimi' kullanıcı tarafından doğrudan fark edilen bir ürün hatasıdır; CPU israfı (denoiser tüm sese uygulanır) kabul edilebilir çünkü model kapsama alanı tüm akış zaten."
  - `pipeline/README.md` Claude Code'a hatırlatma: "Konuşmacı seçimi modülü `BaseSpeakerSelector` arayüzünden türeyecek (load + process(audio, vad_segments) -> selected_audio); ileride yazılacak."
- **Acceptance criteria:**
  - [ ] `tasarim_dokumani.md` 4.4 bölümünü içerir; "Sıradaki" listesinde "Pipeline sıralaması: Akış 1 (önce denoise)" notu var
  - [ ] `pipeline/` klasörü oluşturulmuş, içinde README ve boş `__init__.py` var
- **Out of scope:** Konuşmacı seçimi modülünü yazmak (bu sonraki turun konusu)
- **Depends on:** None (paralel uygulanabilir, Task 1-5'ten önce de sonra da fark etmez)

---

### Task 7 — README ve çalıştırma talimatları güncelle

- **Why:** Yeni dataset indirme, profile sistemi, HTML raporu ortaya çıktı. Yeni bir kullanıcı (veya 3 ay sonraki Selman) hangi komutla ne yapılır anlamalı.
- **Files to touch:**
  - `README.md` (new veya update) — proje kök README
  - `tasarim_dokumani.md` — yol haritası güncelle
- **Approach:**
  - README'de şu akış yazılı olsun:
    1. Kurulum: `pip install -r requirements.txt`
    2. Veri indirme: `python -m scripts.download_data`
    3. Hızlı test: `python -m scripts.bench_synthetic --profile s_smoke`
    4. Tam koşum: `python -m scripts.bench_synthetic --profile s_quick`
    5. Raporu açma: `output/.../report.html` tarayıcıda
  - Tasarım dokümanı "Tamamlanan" listesinde yeni eklenen şeyleri işaretle, "Sıradaki" listesinde konuşmacı seçimi vurgulanır
- **Acceptance criteria:**
  - [ ] README.md sıfırdan bilgisayara kurulan biri için adım adım komut listesi içerir
  - [ ] tasarim_dokumani.md güncel
- **Out of scope:** API dokümanı; örnek notebooks
- **Depends on:** Task 1-5 tamamlandıktan sonra

---

## 4. Gelecek hedefleri (bu turun KONUSU DEĞİL, kaydedilsin)

Aşağıdaki maddeler bu turun planında YOK. Hatırlatma için yazıldı, gelecek turlarda ele alınacak.

- **Konuşmacı seçimi modülü** (VAD + RMS) — bir sonraki tur. `pipeline/speaker_selector.py` ve `BaseSpeakerSelector` ile.
- **Multi-agent değerlendirme sistemi** — Selman'ın fikri: deney sonuçlarını ikinci bir LLM "peer review"-vari değerlendirir, üçüncü bir LLM (sen?) ikisi arasında hakem rolü oynar. Otomatik kalite kontrol katmanı; model seçim kararını daha güvenilir hale getirir. Geçici isim: "Agentic Evaluation Loop".
- **Otomatik kendini iyileştirme döngüsü** — Selman'ın fikri: sistem belirli bir senaryoda zayıf çıkıyorsa hyperparameter veya model seçimini bizzat kendi ayarlar (örn. SNR < 0'da otomatik daha güçlü modele geçer).
- **Canlı mikrofon pipeline** — real-time test, streaming inference.
- **ONNX/TFLite dönüşüm** — kazanan modelin mobile için export'u.
- **Android tarafı ölçümleri** — gerçek cihazda RTF, RAM, latency.

---

## 5. Handoff Protocol

1. **Claude (chat) → GitHub:** bu dosya `master`'a commit edilir.
2. **Selman → Claude Code:** "şunları yap" der.
3. **Claude Code:** sırayla taskları uygular, her birini ayrı commit, acceptance criteria geçerse devam.
4. **Selman + Claude (chat):** Task 5 sonundaki HTML raporu birlikte yorumlar, hipotez sonuçlarını değerlendirir.
5. Bir sonraki turun planlamasına (konuşmacı seçimi) Claude (chat) bu dosyayı arşivler, yeni Plan bölümünü doldurur.
