# `pipeline/` — Akış 1: Önce Denoise, Sonra Konuşmacı Seçimi

Bu klasör nihai uygulama pipeline'ını barındırır. Şu an **placeholder** — modüller
sonraki turda yazılacak. Mimari karar ise şimdi sabitlendi ki "denoise mi önce,
VAD mı önce" tartışması tekrar açılmasın.

## Karar: Akış 1 (önce denoise, sonra speaker selection)

```
mic / .wav
   │
   ▼
┌────────────────────┐
│  BaseDenoiser      │  → kazanan model (DeepFilterNet öncü)
│  process(audio)    │
└────────────────────┘
   │
   ▼
┌────────────────────┐
│  BaseSpeakerSelect │  → VAD + RMS tabanlı baskın konuşmacı seçici
│  process(audio,    │     (ileride yazılacak)
│          vad_segs) │
└────────────────────┘
   │
   ▼
output
```

### Neden Akış 1?

1. **VAD temiz ses üzerinde daha güvenilir karar verir.** Klavye/trafik gürültüsü
   VAD'i de yanıltır; denoise sonrası VAD'in false-positive oranı ciddi düşer.
2. **"Yanlış konuşmacı seçimi" en görünür kullanıcı hatasıdır.** Hedef cihaz
   telsiz benzeri; iletilen sesin doğru kişi olması, gürültüden temiz olmasından
   daha kritik. Bu yüzden seçim adımının kararlılığını yükseltmek gerekir.
3. **CPU israfı tolere edilebilir.** Denoiser'ı tüm akışa uyguluyoruz; sadece
   "konuşmacı tespit edilen pencerelere" uygulamak optimizasyon olur ama
   model kapsama alanı (RTF < 0.1) yeterince düşük.

### Alternatif (Akış 2): Önce VAD, sadece konuşma bölgelerinde denoise

Reddedildi çünkü:
- VAD gürültülü seste yanıltıcı; konuşma bölgelerini eksik tespit edebilir
- Pencere geçişlerinde "denoise edilmiş ↔ ham" sınırı duyulabilir bozulmalar yaratır
- Speaker selection adımı yine gürültülü ham sinyalle çalışmak zorunda kalır

## İleride yazılacak (sonraki tur)

```python
# pipeline/base.py  — bu dosya henüz yazılmadı, taslak şu olacak:
#
# from abc import ABC, abstractmethod
# import numpy as np
#
# class BaseSpeakerSelector(ABC):
#     """Birden fazla konuşmacılı sinyalden baskın olanı seçen modül.
#
#     `load()` ağırlık/kaynak yükler (örn. Silero VAD modeli).
#     `process(audio, vad_segments=None)` denoise edilmiş tek-kanal sesi alır;
#     baskın konuşmacının dışındaki bölgeleri bastırır.
#
#     Args:
#       audio: float32 mono (N,), 16 kHz, denoise sonrası
#       vad_segments: opsiyonel, dış VAD çıktısı [(start_s, end_s), ...]
#                     None ise dahili VAD kullanılır
#
#     Returns:
#       float32 mono (N,), zayıf konuşmacılar bastırılmış
#     """
#     sample_rate = 16000
#     name = "base_selector"
#
#     @abstractmethod
#     def load(self): ...
#
#     @abstractmethod
#     def process(self, audio: np.ndarray,
#                 vad_segments: list[tuple[float, float]] | None = None) -> np.ndarray: ...
```

Aday yöntemler (bir sonraki planlama turunda kararlaştırılacak):
- **Silero VAD** + RMS pencereleme — modern, hafif (ONNX 2 MB)
- **WebRTC VAD** + RMS — Google WebRTC C kütüphanesi, çok hafif ama eski
- **py-webrtcvad** — yukarıdakinin Python binding'i, mobil için ideal
- Saf enerji eşiklemesi — VAD yok, sadece RMS — en hafif ama gürültüden etkilenir

## Bench üzerinden test

`scripts/bench_synthetic.py` şu an sadece denoiser'a odaklı. İleride
`pipeline.run_full(audio, denoiser, selector)` fonksiyonu yazılacak ve
multi-speaker karışımlar (iki temiz konuşma + arka plan gürültü) üzerinde
end-to-end test yapılabilecek.
