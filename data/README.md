# Dataset References

Bu klasör benchmark için gereken **temiz konuşma** + **gürültü** ses dosyalarını barındırır. Tüm sesler **16 kHz mono float32 PCM_16 WAV** olarak normalize edilmiş kaydedilir.

İndirme: `python -m scripts.download_data` (proje kökünden).

## Klasör organizasyonu

```
data/
├── clean/
│   ├── en/    # VCTK İngilizce konuşma (10 dosya: 5M + 5F konuşmacı)
│   │   ├── spk{p225..p234}_utt001.wav
│   │   └── ...
│   └── tr/    # Common Voice Türkçe (10 dosya, kalite filtresi geçen)
│       ├── spk{client_id_short}_utt{N}.wav
│       └── ...
└── noise/     # DEMAND 7 sahne (her sahnenin ch01.wav'ı)
    ├── TBUS/ch01.wav
    ├── TCAR/ch01.wav
    ├── TMETRO/ch01.wav
    ├── SCAFE/ch01.wav
    ├── SPSQUARE/ch01.wav
    ├── OOFFICE/ch01.wav
    └── NPARK/ch01.wav
```

## Kaynaklar ve lisanslar

### 1. VCTK Corpus (İngilizce konuşma)
- **URL:** https://datashare.ed.ac.uk/handle/10283/3443
- **HuggingFace mirror:** `CSTR-Edinburgh/vctk`
- **Lisans:** Open Data Commons Attribution License (ODC-By)
- **Bizim alt küme:** 10 konuşmacı (5M + 5F, çeşitli aksanlar), her birinden 1 cümle = 10 dosya
- **Hak sahibi:** University of Edinburgh, Centre for Speech Technology Research

### 2. Common Voice Turkish (Türkçe konuşma)
- **URL:** https://commonvoice.mozilla.org/tr
- **HuggingFace:** `mozilla-foundation/common_voice_17_0` (config: `tr`)
- **Lisans:** CC0 1.0 (Public Domain)
- **Bizim alt küme:** `up_votes ≥ 2` ve `down_votes == 0` filtresi geçen 10 örnek, konuşmacı çeşitliliği için her örneğin `client_id`'si farklı.
- **GATED:** Önce https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 sayfasında "Agree" + lokal `huggingface-cli login`.
- **Hak sahibi:** Mozilla Foundation + bağışçı konuşmacılar.

### 3. DEMAND (gürültü)
- **URL:** https://zenodo.org/record/1227121
- **Lisans:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Bizim alt küme:** 7 sahne × ch01.wav (mikrofon dizi içinden tek kanal, ~5 dk)
  - **TBUS** — otobüs içi (trafik araç gürültüsü)
  - **TCAR** — araç içi (motor + yol gürültüsü)
  - **TMETRO** — metro içi
  - **SCAFE** — kafe (kalabalık + müzik)
  - **SPSQUARE** — halk meydanı (kalabalık konuşma)
  - **OOFFICE** — ofis (klima + fan + uğultu)
  - **NPARK** — park (rüzgar + doğa sesleri)
- **Hak sahibi:** Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent.

## Dosya formatı doğrulama

```python
import soundfile as sf
info = sf.info("data/noise/TCAR/ch01.wav")
assert info.samplerate == 16000
assert info.channels == 1
```

## Yeniden indirme

```bash
# Sadece bir kaynak
python -m scripts.download_data --only vctk
python -m scripts.download_data --only cv_tr
python -m scripts.download_data --only demand

# Mevcut dosyaları üzerine yaz
python -m scripts.download_data --force
```

## Bilinen sorunlar

- **Common Voice gated:** HuggingFace hesabı + dataset için "Agree" + `huggingface-cli login` gerekir. Script gated hatasını yakalar ve devam eder; ancak `data/clean/tr/` boş kalır.
- **DEMAND zip büyüklüğü:** Sahne başına ~80-200 MB. Toplam ~700 MB-1 GB. Yavaş bağlantıda indirme süresi uzun olabilir; script timeout=120 sn.
- **SSL sertifika hataları (Windows):** Bazı kurumsal ağlarda `CRYPT_E_NO_REVOCATION_CHECK` görülebilir. Çözüm: `git config http.schannelCheckRevoke false` (git için) veya kurumsal CA bundle kurulumu.
