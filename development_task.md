# Development Task — Handoff Document

> **Purpose.** Bu dosya iki Claude arasında bir köprü:
> - **Claude (chat / claude.ai)** planlamayı yapar, bu dosyayı yazar.
> - **Claude Code (IDE)** dosyayı okur, taskları sırayla uygular.
>
> **Aktif tur: Tur 2 — Pre-built mix üretici ve benchmark adaptasyonu.**
> **Önceki tur (Tur 1) tamamlandı** ve aşağıda Bölüm 6'da arşivlenmiştir.

---

## 1. Tur 2'nin amacı ve gerekçesi

Selman, mevcut sentetik veri üretiminden tatmin değil. Mix'ler şu anda `bench_synthetic.py` içinde **bellekte üretiliyor ve doğrudan modele gönderiliyor** — yani diskte saklanmıyor. Sonuç olarak kullanıcı, gürültünün doğru SNR'da eklendiğinden emin olamıyor, mix'leri kulakla doğrulayamıyor.

**İstenen yeni akış:**

1. Mix'ler **ayrı bir adımda önceden üretilir** ve `input_data/{profile}/` altına yazılır.
2. Kullanıcı `input_data/{profile}/` klasöründen birkaç dosyayı rastgele dinleyerek "spot check" yapar.
3. Spot check tamamsa benchmark çalıştırılır; benchmark **pre-built mix'leri kullanır**, bellekte üretmez.
4. Pre-built mix yoksa benchmark mevcut akışla (bellekte) devam eder ama önce kullanıcıya sorar.

Bu sayede:
- Mix doğruluğu kullanıcı tarafından önden doğrulanabilir.
- Benchmark koşumları arası **aynı mix dosyaları kullanılır** — tekrar üretilebilirlik artar.
- Profil tabanlı yeniden kullanım: `s_quick` üretilmişse, `s_smoke` (alt küme) kendi dosyalarını sıfırdan üretmez, var olanlardan kopyalar.

## 2. Bu turun bağlamı

### 2.1 Mevcut durum
- Tur 1 tamamlandı. Bütün benchmark altyapısı çalışıyor (dataset indirme, profile sistemi, akıllı örnek seçici, anomali yakalama, HTML rapor).
- Hipotez kayıt altında, otomatik test ediliyor (H1-H4).
- `bench_synthetic.py` clean + noise klasörlerinden mix'leri bellekte üretiyor.

### 2.2 Hedeflenen son durum
- `scripts/build_mixes.py` — ayrı bir CLI, mix'leri diske yazar, manifest üretir.
- `bench_synthetic.py` pre-built mix tespit ederse onları kullanır; yoksa kullanıcıya sorup eski akışı çalıştırır.
- F5/konsol çalıştırma deneyimi değişmez: her iki script konfigürasyonu interaktif sorar.

### 2.3 Geriye dönük uyumluluk
- `data/clean/` ve `data/noise/` ham hammadde olarak kalır (Tur 1 Task 1 ile indirildi).
- Eski test yolu (clean + noise üzerinden bellekte mix) **kaldırılmaz**, fallback olarak kalır.
- Mevcut profil sistemi (`s_quick`, `s_smoke`, `m_medium`) aynen kullanılır.

---

## 3. Karar özeti (binding)

| Konu | Karar |
|---|---|
| **Mix klasörü** | `input_data/{profile_name}/` — örn. `input_data/s_quick/` |
| **Mix dosya formatı** | `{lang}_{clean_id}__{noise_scene}__snr{XX}.wav` |
| **Mix kapsamı** | profile parametrelerinden hesaplanır: `max_pairs × scenes × snrs` (rep yok — aynı kombinasyon bir kez yazılır) |
| **Manifest** | `input_data/{profile}/manifest.csv` — her satır: clean_path, noise_path, noise_offset_sec, target_snr_db, achieved_snr_db, duration_sec, expected_filename |
| **Profil arası yeniden kullanım** | Aynı (clean, noise, SNR) kombinasyonu başka bir profilde önceden üretilmişse **kopya** alınır, yeniden üretilmez |
| **build_mixes CLI** | İnteraktif (profil sorar) + flag'ler: `--profile`, `--force`, `--limit` |
| **bench_synthetic adaptasyonu** | Manifest varsa "pre-built kullanayım mı? [E/h]"; yoksa "bellekte üreteyim mi? [E/h]" |
| **Idempotency** | İkinci çağrıda mevcut dosyaları yeniden üretmez, "X/Y zaten var, atlanıyor" mesajı verir |
| **Spot check otomasyonu** | Bu turda **yapılmıyor**; LLM-tabanlı doğrulama gelecek hedefler listesinde kalır |

---

## 4. Tur 2 Task Listesi

Tasklar sırayla uygulanacak. Her task acceptance criteria'ları geçince commit edilir.

---

### Task T2-1 — `build_mixes.py` scripti

- **Why:** Mix'leri önceden diske yazmak ve manifest üretmek için yeni CLI lazım. Mevcut `benchmark/mixer.py` sadece fonksiyon (bellekte mix); bu task onu sarmalayan disk-yazıcı script ekler.

- **Files to touch:**
  - `scripts/build_mixes.py` (new) — ana CLI
  - `benchmark/mix_manifest.py` (new) — manifest okuma/yazma yardımcıları
  - `.vscode/launch.json` — yeni Run konfigürasyonu

- **Approach:**
  - **CLI argümanları:**
    - `--profile NAME` — profil adı; verilmezse interaktif olarak sorar (`s_quick` / `s_smoke` / `m_medium`)
    - `--force` — mevcut dosyaları üzerine yaz, idempotent davranışı atla
    - `--limit N` — test için, sadece ilk N kombinasyonu üret
  - **Profil parametreleri:** `scripts/profiles.py`'den okur (zaten var). Profil dict'inden `clean_dir`, `noise_dir`, `snrs`, `max_pairs` alınır. `n_repeats` mix üretiminde **kullanılmaz** (her kombinasyon bir kez yazılır).
  - **Klasör hazırlığı:** `input_data/{profile}/` yoksa oluştur.
  - **Mix kombinasyon listesi:**
    - `clean_dir` altındaki dil klasörlerini gez (`en/`, `tr/`).
    - Her dilden eşit pay alarak `max_pairs` clean dosya seç (örn. max_pairs=20 → 10 en + 10 tr). Deterministik seçim için seed=42.
    - `noise_dir` altındaki sahne klasörlerini gez (DEMAND'ın 7 sahnesi). Her sahnenin `ch01.wav`'ı kullanılır.
    - Tüm (clean, noise, snr) kombinasyonları: `max_pairs × 7 × 5` — örn. s_quick için 700 dosya.
  - **Her kombinasyon için:**
    - `audio_io.file_io.load_audio` ile clean'i yükle (16 kHz mono).
    - Noise'u yükle, clean'in uzunluğuna göre **rastgele offset'ten kırp** (seed deterministik). Offset manifestte kaydedilir.
    - `benchmark.mixer.mix_at_snr` ile hedef SNR'a karıştır.
    - Achieved SNR'u ölç (gerçekleşen SNR ile hedef arası fark; mixer 0.1 dB tolerans hedefliyor).
    - Hedef dosya yolu: `input_data/{profile}/{lang}_{clean_id}__{noise_scene}__snr{XX}.wav`. `snr{XX}` formatı: -5 → `-05`, 0 → `00`, 10 → `10` (iki haneli, sayısal sıralanabilir).
    - `save_audio` ile yaz.
  - **Profil arası yeniden kullanım:**
    - Mix yazmadan önce **diğer profil klasörlerinde** aynı dosya adına sahip bir dosya var mı bak (basit dosya adı eşitliği yeterli; clean+noise+snr kombinasyonu dosya adında zaten var).
    - Varsa kopyala (`shutil.copy2`), tekrar üretme.
    - Hangi profilden kopyalandığını manifest'in `source` sütununda işaretle (`generated` veya `copied_from:s_quick`).
  - **Manifest yazımı:**
    - `manifest.csv` — UTF-8, başlık satırı, virgülle ayrılmış.
    - Sütunlar: `idx`, `lang`, `clean_id`, `clean_path`, `noise_scene`, `noise_path`, `noise_offset_sec`, `target_snr_db`, `achieved_snr_db`, `duration_sec`, `mix_filename`, `source`.
  - **Idempotency:**
    - Çalıştırma başında mevcut `manifest.csv` varsa oku, hangi `mix_filename`'lar var listele.
    - Her kombinasyon için: dosya **ve manifest satırı** mevcutsa atla. Eksik olan satırlar için tekrar üret.
    - `--force` ile bu kontrolü atla.
  - **İlerleme:** tqdm ile progress bar. Toplam sayı = profil için tüm kombinasyon adedi.
  - **Konsol özeti:** sonda yazılan/atlanan/kopyalanan dosya sayısı + manifest yolu.

- **Acceptance criteria:**
  - [ ] `python -m scripts.build_mixes --profile s_smoke` çalıştırınca `input_data/s_smoke/` altında profil parametrelerine uygun sayıda wav (`max_pairs × scenes × snrs`) ve `manifest.csv` oluşur
  - [ ] Manifest'in her satırı diskte gerçek bir dosyaya işaret eder; eksik dosya yoktur
  - [ ] `python -m scripts.build_mixes` (argümansız) profil seçim sorusu açar; bilinmeyen profilde anlamlı hata verir
  - [ ] İkinci çağrıda "X/Y zaten var, atlanıyor" mesajıyla 5 saniyenin altında biter
  - [ ] `--force` flag'i ile tüm dosyalar üzerine yazılır
  - [ ] `--limit 10` ile sadece 10 mix üretilir
  - [ ] `s_smoke` üretildikten sonra `s_quick` çalıştırılırsa, `s_smoke`'taki dosyalar (eğer dosya adı eşleşirse) kopyalanır, yeniden üretilmez; manifest `source` sütununda işaretlenir
  - [ ] Achieved SNR ile target SNR arası fark < 0.5 dB (mixer hatası yoksa)

- **Out of scope:**
  - Kullanıcıya mix'leri dinletme arayüzü (manuel klasör açma yeterli)
  - Otomatik mix doğrulama (LLM tabanlı gözlemci — gelecek hedef)
  - Profil dışı SNR seviyeleri (sabit: profil ne diyorsa o)

- **Depends on:** Tur 1 Task 1 (dataset indirme tamamlanmış olmalı)

---

### Task T2-2 — `bench_synthetic.py` pre-built mix desteği

- **Why:** Pre-built mix'ler hazırsa benchmark onları kullanmalı; yoksa kullanıcıya sorup mevcut bellek-mix akışına düşmeli.

- **Files to touch:**
  - `scripts/bench_synthetic.py` — pre-built tespit + manifest okuma + ses yükleme yolu
  - `benchmark/mix_manifest.py` — Task T2-1'de yaratılan modül, burada `load_manifest()` fonksiyonu kullanılır

- **Approach:**
  - **Akış değişikliği:** Profil seçimi sonrası, mevcut akışa şu kontrol eklenir:
    ```
    1. input_data/{profile}/manifest.csv var mı?
       Evet → "Pre-built mix bulundu ({N} dosya). Bunları kullanayım mı? [E/h]" sor.
              E (varsayılan) → manifest'i oku, her satır için mix dosyasını yükleyip modele ver. Bellekte mix yapma.
              h → bellekte üret (mevcut akış).
       Hayır → "Pre-built mix yok. Bellekte üreteyim mi? [E/h]" sor.
              E (varsayılan) → mevcut akış.
              h → çıkış kodu 0 ile bilgilendirici mesaj ("Önce 'python -m scripts.build_mixes --profile {profile}' çalıştırın").
    ```
  - **CLI flag:** `--use-prebuilt {auto, yes, no}`:
    - `auto` (varsayılan): interaktif soru
    - `yes`: pre-built varsa kullan, yoksa hata
    - `no`: pre-built varsa bile bellekte üret
  - **Pre-built kullanım yolu:**
    - `load_manifest("input_data/{profile}/manifest.csv")` çağrısı satırları döner
    - Her satır için:
      - `mix_filename` ile `noisy = load_audio(...)`
      - Clean referansı: `clean_path` sütununu kullan, `clean = load_audio(...)`
      - Sahne ve SNR doğrudan manifest'ten okunur, hesaplama yok
    - Sonra her model bu (noisy, clean) çiftine `n_repeats` kez uygulanır (rep yine bench içinde, ama mix bir kere yüklenir, modeller arası ortak kalır)
  - **Performans optimizasyonu:** Mix dosyaları bellekte cache'lenir; aynı mix farklı modeller için tekrar diskten okunmaz. Bu zaten mevcut akışta benzer şekilde işliyor; cache yapısı korunur.
  - **Hata durumu:** Manifest var ama bazı dosyalar eksik → exit, hangi dosyaların eksik olduğunu listele, `build_mixes` ile tamamla mesajı.

- **Acceptance criteria:**
  - [ ] `python -m scripts.bench_synthetic --profile s_smoke` (manifest yokken) "pre-built yok, bellekte üreteyim mi" sorar
  - [ ] `build_mixes --profile s_smoke` çalıştırıldıktan sonra `bench_synthetic --profile s_smoke` "pre-built var, kullanayım mı" sorar
  - [ ] `--use-prebuilt yes` ile etkileşimsiz pre-built kullanımı çalışır
  - [ ] `--use-prebuilt no` ile mevcut bellek-mix akışı çalışır
  - [ ] Pre-built ile çalıştırma ile bellek-mix ile çalıştırma **aynı sayıda ölçüm** üretir (manifest dosya sayısı × n_repeats × n_models)
  - [ ] Pre-built ile çalıştırma ile bellek-mix ile çalıştırma sonuçları (PESQ, STOI, SI-SDR ortalamaları) **yaklaşık aynı** olur — birebir aynı değil çünkü noise offset random olabilir (mixer seed kullanıyorsa birebir, kullanmıyorsa yakın)
  - [ ] Manifest'te listelenen ama diskte olmayan dosya varsa anlamlı hata: "Manifest 700 dosya bekliyor ama 698 bulundu. Eksikler: ..."
  - [ ] Mevcut tüm Tur 1 acceptance criteria'ları geçmeye devam eder (regresyon yok)

- **Out of scope:**
  - Manifest'in başka bir formatta (JSON, Parquet) desteklenmesi
  - Bellek-mix akışını tamamen kaldırmak (fallback olarak kalır)

- **Depends on:** Task T2-1

---

### Task T2-3 — F5/Run konfigürasyonu ve README

- **Why:** Yeni `build_mixes` adımı kullanıcı akışının başına eklendi. Yeni gelen biri (veya 3 ay sonraki Selman) bu adımı atlayıp bench koştuğunda kafası karışmasın.

- **Files to touch:**
  - `.vscode/launch.json` — yeni konfigürasyon
  - `README.md` — Hızlı Başlangıç bölümü güncelle
  - `tasarim_dokumani.md` — Yol Haritası "Tamamlanan" bölümüne ekleme

- **Approach:**
  - `.vscode/launch.json` yeni konfigürasyonlar:
    - **build_mixes (interactive profile)** — `python -m scripts.build_mixes`, profil sorusu açar
    - **build_mixes (profile: s_quick)** — direkt s_quick üretir
    - **build_mixes (profile: s_smoke — fast)** — direkt s_smoke üretir
  - README.md "Hızlı Başlangıç" bölümünde adım sırası:
    1. Kurulum
    2. Veri indirme (`scripts.download_data`)
    3. **(YENİ) Mix üretimi (`scripts.build_mixes --profile s_smoke`)** — kullanıcı klasörü açıp birkaç dosyayı dinleyerek SNR doğruluğunu kontrol eder
    4. Smoke test (`scripts.bench_synthetic --profile s_smoke`)
    5. Tam koşum (`scripts.bench_synthetic --profile s_quick`)
    6. Raporu açma
  - README.md'de yeni bir "Mix doğrulama" mini bölümü: kullanıcıya "klasörden 3-5 dosya rastgele dinleyin, gürültü doğru seviyede karışmış mı emin olun" talimatı.
  - `tasarim_dokumani.md` Tamamlanan listesine yeni satırlar:
    - `[x] Pre-built mix üretici (build_mixes.py) — input_data/{profile}/ altında manifest + wav`
    - `[x] bench_synthetic pre-built mix desteği (--use-prebuilt auto/yes/no)`
    - `[x] Profil arası mix yeniden kullanımı (kopya optimizasyonu)`

- **Acceptance criteria:**
  - [ ] `.vscode/launch.json`'da build_mixes için en az 2 konfigürasyon (interaktif + direkt s_smoke)
  - [ ] README.md adımları yeni 5 numaralı (yenisiyle 6 numaralı) akışı yansıtır
  - [ ] `tasarim_dokumani.md` güncel
  - [ ] Bir başkası README.md'yi takip ederek baştan sona koşumu hatasız tamamlayabilir

- **Out of scope:** Çeviri (İngilizce README); video tutorial; ek diagram

- **Depends on:** Task T2-1, Task T2-2

---

## 5. Gelecek hedefleri (Tur 2'nin KONUSU DEĞİL, hatırlatma)

Bu maddeler Tur 2'de yapılmıyor. Selman'ın talebine göre bu listeye eklendi/güncellendi.

- **Mix doğrulama için LLM gözlemci** — Selman spot check yapacak ama "tüm 700 dosyayı dinleyemem" dedi. İleride bir LLM-tabanlı veya sinyal-tabanlı otomatik doğrulayıcı: her mix'in SNR'ı, kırpılma, format hatalarını otomatik kontrol eder, anormal olanları işaretler.
- **Konuşmacı seçimi modülü (VAD + RMS)** — Tur 3'ün konusu. Pipeline mimarisi: Akış 1 (önce denoise → sonra seçim). Aday VAD'ler: Silero / WebRTC / saf enerji.
- **Multi-agent değerlendirme sistemi** — Bench sonuçlarını ikinci bir LLM "peer review"-vari değerlendirir.
- **Otomatik kendini iyileştirme döngüsü** — Sistem zayıf çıkan senaryolarda hyperparameter/model seçimini bizzat ayarlar.
- **Canlı mikrofon pipeline** — Real-time test, streaming inference.
- **ONNX/TFLite dönüşüm** — Kazanan modelin mobile için export'u.
- **Android tarafı ölçümleri** — Gerçek cihazda RTF, RAM, latency.

---

## 6. Tur 1 — Arşiv (TAMAMLANDI)

> Bu bölüm Tur 1'in özetidir; tasklar tamamlandı, kod tabanına işlendi.

### 6.1 Tur 1'in odağı
Kapsamlı çok-değişkenli denoiser benchmark altyapısının veri ve raporlama yönleriyle tamamlanması. Konuşmacı seçimi modülü Tur 1'in konusu değildi.

### 6.2 Tur 1'in araştırma sorusu
"Yedi denoiser modelinin farklı gürültü tiplerinde, farklı SNR seviyelerinde, hem Türkçe hem İngilizce konuşmada karşılaştırmalı performansı nedir; hangi(leri) mobil/telsiz hedefimiz için en uygun adaydır?"

### 6.3 Tur 1'in hipotezi
Bu hipotez sonuçlar görülmeden önce kayıt altına alındı; rapor onayı/çürütmesini otomatik test ediyor.

> Klasik/hafif yöntemler (spectral_subtraction, RNNoise) düşük SNR ve konuşmacı gürültüsü gibi karmaşık ortamlarda zorlanır. DL modelleri (DeepFilterNet, Demucs varyantları) fan/motor/uğultu gibi sürekli gürültülerde daha iyi sonuç verir. MetricGAN+ bazı düşük SNR senaryolarında gürültüyü agresif bastırırken konuşma doğallığını bozar. Kafe/kalabalık konuşma gibi hedef konuşmaya benzeyen gürültülerde tüm modeller zorlanır; model farkları en net burada ortaya çıkar.

Hipotez 4 alt-iddiaya ayrıştırıldı (otomatik test için):
- **H1.** SNR ≤ 0 dB'de spectral_subtraction + rnnoise'ın ortalama PESQ'si, DL modellerinin ortalama PESQ'sinden ≥ 0.3 düşüktür.
- **H2.** Sürekli/sabit gürültü sahnelerinde (DKITCHEN, OOFFICE, TCAR, NPARK) DFN ve Demucs ailesinin ortalama PESQ'si, klasik yöntemlerden ≥ 0.4 yüksektir.
- **H3.** MetricGAN+ çıkışlarında, en az iki SNR seviyesinde ortalama RMS düşüşü > 8 dB veya HF Energy Ratio < 0.0010.
- **H4.** Kafe/kalabalık konuşma sahnesinde (SCAFE, SPSQUARE) tüm modellerin PESQ ortalamaları yakın (max−min < 0.4).

### 6.4 Tur 1 tasklar (özet)
- ✅ **T1-1** Dataset indirme scripti (VCTK + Common Voice TR + DEMAND 7 sahne)
- ✅ **T1-2** Profile sistemi (`s_quick`, `s_smoke`, `m_medium`)
- ✅ **T1-3** Akıllı örnek seçici + `--save-strategy` (`benchmark/sampling.py`)
- ✅ **T1-4** Anomali yakalama (4 detektör — `benchmark/anomaly.py`)
- ✅ **T1-5** HTML rapor üreticisi (`benchmark/html_report.py` + `spectrogram.py` + `hypothesis.py`)
- ✅ **T1-6** Pipeline mimari kararı dokümantasyonu (Akış 1 — önce denoise)
- ✅ **T1-7** README ve çalıştırma talimatları güncellendi

### 6.5 Tur 1 kararları
| Konu | Karar |
|---|---|
| **Veri stratejisi** | Strateji B: VCTK alt küme + Common Voice TR + DEMAND 7 sahne |
| **Senaryo** | S: 20 temiz × 7 gürültü × 5 SNR × 3 tekrar = 2100 işlem/model |
| **SNR seviyeleri** | −5, 0, 5, 10, 15 dB |
| **DEMAND sahneleri** | TBUS, TCAR, TMETRO, SCAFE, SPSQUARE, OOFFICE, NPARK |
| **Temiz konuşma** | 10 İngilizce klip + 10 Türkçe klip |
| **Rapor formatı** | HTML tek dosya, embedded audio + spektrogram + otomatik hipotez testi |
| **Dinleme galerisi** | 2 sahne (SCAFE, TCAR) × 2 SNR (−5, 5) × 7 model = 28 örnek |
| **Pipeline akışı** | Akış 1 — önce denoise → sonra konuşmacı seçimi |

---

## 7. Handoff Protocol

1. **Claude (chat) → GitHub:** Bu dosya `master`'a commit edilir.
2. **Selman → Claude Code:** "Tur 2 tasklarını sırayla yap" der.
3. **Claude Code:** T2-1, T2-2, T2-3'ü sırayla uygular, her birini ayrı commit, acceptance criteria geçerse devam.
4. **Selman:** Task T2-1 tamamlandıktan sonra `input_data/s_smoke/` klasörünü açıp birkaç dosyayı dinleyerek spot check yapar. Mix'lerden tatmin olmazsa parametre ayarı için Claude (chat)'e geri döner.
5. **Selman + Claude (chat):** T2-3 tamamlandıktan sonra bench koşulur, HTML raporu birlikte yorumlanır.
6. Bir sonraki tur (Tur 3 — konuşmacı seçimi) planlamasına Claude (chat) bu dosyayı arşivler, yeni Plan bölümünü doldurur.
