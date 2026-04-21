# Gürültü Bastırma Modelleri — Karşılaştırma Raporu

**Deney tarihi:** 2026-04-21, 15:02
**Giriş sesi:** 10 saniyelik PC mikrofon kaydı (16 kHz mono, 16-bit PCM)
**Donanım:** CPU-only, torch 2.3.1+cpu

---

## 1. Raporun Amacı

Bu rapor, ses izolasyonu projesinde değerlendirilen yedi denoiser modelinin **aynı ses girişi üzerinde** yan yana karşılaştırılmasını sunar. Amaç, mobil (Android) hedefi göz önüne alarak **hız–boyut–kalite üçgeninde** hangi modelin aday olabileceğini belirlemektir. Raporda hem benchmark rakamları (RTF, RAM, boyut, parametre sayısı) hem de çıktı seslerinin sinyal-seviyesi objektif analizi (RMS, gürültü tabanı, spektral centroid, yüksek frekans enerji oranı) birlikte sunulmuştur.

## 2. Karşılaştırmaya Dahil Modeller

| Kısa ad | Mimari ailesi | Domain |
|---|---|---|
| spectral_subtraction | Klasik DSP | Frekans |
| rnnoise | DSP + küçük RNN | Bant-temelli |
| deepfilternet | CNN + GRU, iki aşamalı | Frekans |
| demucs_dns48 | Encoder-Decoder + LSTM | Zaman |
| demucs_dns64 | Encoder-Decoder + LSTM | Zaman |
| demucs_master64 | Encoder-Decoder + LSTM | Zaman |
| metricgan_plus | GAN tabanlı | Frekans |

## 3. Ölçülen Metriklerin Tanımı

### 3.1 Performans Metrikleri (Mobil Uygunluk için)

**RTF (Real Time Factor).** İşleme süresi / ses süresi. 1'den küçükse model sesi **gerçek zamanlı işleyebilir** (yani 1 saniyelik sesi 1 saniyeden kısa sürede temizler). Örn. RTF=0.1, modelin 10× gerçek zamanlı olduğu anlamına gelir. Mobil cihazda aynı modelin 3–5 kat yavaşlayacağı hesaba katılırsa, PC üzerinde RTF ≲ 0.2 olması mobil real-time için güvenli bir eşiktir.

**process_time_s.** Modelin `process()` çağrısı üzerindeki mutlak süre (saniye). RTF'in payıdır; kullanıcı açısından algılanan "gecikme" buna eşdeğerdir.

**load_time_s.** Modelin `load()` aşamasında ağırlıkları/durumunu kurma süresi. Uygulama açılışında bir kere ödenir; sonraki çağrıları etkilemez ama ilk çalışma deneyimi açısından önemlidir.

**ram_load_mb / ram_process_mb.** Modelin yüklendiğinde ve ses işlediğinde bellek kullanımındaki artış. Gömülü sistemlerde ve mobilde sıkı kısıttır. *Not: Bu değerler `psutil` ile "işlem sonrası − öncesi" farkı olarak ölçüldü; işlem anındaki peak belleği tam olarak yansıtmaz, kabaca bir gösterge sağlar. Bazı satırlarda eksi değer görünmesi garbage collection tarafından daha önceki modellerin belleğinin temizlenmesinden kaynaklanır, model özelinde "gerçek kullanım" değildir.*

**model_size_mb.** Model ağırlıklarının diskteki / bellekteki boyutu (parametre sayısı × float boyutu). Doğrudan APK büyüklüğüne yansır. Android için 50 MB üstü pratikte sorun olmaya başlar.

**param_count.** Sinir ağının toplam öğrenilebilir parametre sayısı. Model kapasitesinin ve karmaşıklığının göstergesidir; boyutla doğru orantılı ama mimari hakkında ek bilgi verir. *Klasik DSP ve RNNoise için 0 görünüyor — RNNoise'da ağırlıklar PyTorch parametresi olarak değil, C kütüphanesi içinde tutuluyor; bu yüzden bizim ölçüm mantığımız yakalayamıyor.*

### 3.2 Çıktı Kalitesi Metrikleri (Sinyal Analizi)

Objektif PESQ/STOI/SI-SDR metrikleri temiz referans gerektirir ve bu deneyde (gerçek mikrofon kaydı) mevcut değildir. Onun yerine çıktı seslerinin **sinyal-seviyesi özelliklerini** ölçtük:

**RMS (dB).** Sesin ortalama enerjisi. Temizleme sonrası RMS'in çok düşmesi "model ses seviyesini de kırptı" anlamına gelebilir.

**Peak (dB).** Sesteki en yüksek örnek. Normalize çıktılarda 0 dB olur.

**Noise Floor (dB).** Sesin en sessiz bölgelerinin RMS'i — gürültü tabanı tahmini. Ne kadar düşükse, model sessiz anları o kadar temizlemiş demektir.

**Dynamic Range (dB).** RMS − Noise Floor. Konuşma ile zemin arasındaki kontrast; yüksek olması temiz bir gürültü bastırmanın göstergesidir.

**Spectral Centroid (Hz).** Spektrumun "ağırlık merkezi" — sesin parlaklık/boğukluk göstergesi. Temiz konuşmada tipik olarak 1400–1900 Hz civarı. Düşük çıkması → boğuk, yüksek çıkması → cızırtılı/metalik.

**HF Energy Ratio.** 4 kHz üzerindeki enerjinin toplam enerjiye oranı. Yüksek olması yüksek frekanslı gürültü/artefakt var demektir; çok düşük olması ise modelin tüm yüksek frekansı kestiğini (boğuk ses) gösterir.

---

## 4. Ham Benchmark Sonuçları

### Performans Tablosu

| Model | RTF | Süre (s) | Yükleme (s) | Boyut (MB) | Parametre |
|---|---:|---:|---:|---:|---:|
| spectral_subtraction | 0.004 | 0.043 | 0.000 | 0.00 | 0 |
| rnnoise | 0.162 | 1.624 | 2.481 | 0.00* | 0* |
| deepfilternet | 0.043 | 0.430 | 0.273 | 8.15 | 2 135 484 |
| demucs_dns48 | 0.104 | 1.038 | 0.186 | 71.98 | 18 867 937 |
| demucs_dns64 | 0.301 | 3.009 | 0.231 | 127.92 | 33 533 569 |
| demucs_master64 | 0.080 | 0.796 | 0.231 | 127.92 | 33 533 569 |
| metricgan_plus | 0.018 | 0.184 | 1.024 | 7.23 | 1 895 514 |

*RNNoise boyut/parametre ölçümü PyTorch varsayımı yapıyor; C tarafındaki gerçek boyut ~85 KB.

### Kalite Tablosu (Sinyal Analizi)

| Dosya | RMS (dB) | Noise Floor (dB) | Dyn Range (dB) | Centroid (Hz) | HF Ratio |
|---|---:|---:|---:|---:|---:|
| 00_original | −7.70 | −17.71 | 10.0 | 1363 | 0.0037 |
| 01_spectral_subtraction | −10.53 | −34.52 | 24.0 | 1454 | 0.0041 |
| 02_rnnoise | −11.58 | −77.62 | 66.0 | 1364 | **0.0185** |
| 03_deepfilternet | −11.82 | −65.71 | 53.9 | 1427 | 0.0045 |
| 04_demucs_dns48 | −9.67 | −71.39 | 61.7 | 1707 | 0.0040 |
| 05_demucs_dns64 | −9.74 | −77.30 | 67.6 | 1895 | 0.0039 |
| 06_demucs_master64 | −10.02 | −48.56 | 38.5 | 1448 | 0.0041 |
| 07_metricgan_plus | **−19.11** | −44.02 | 24.9 | **1022** | **0.0002** |

Kalın değerler ilgili modelde belirgin sapma gösterir.

---

## 5. Analiz

### 5.1 Gürültü Bastırma Gücü

Orijinalin gürültü tabanı −17.7 dB. Bu yüksek bir değer — stüdyo kaydında −60 dB ve altı beklenir — yani bu gerçek dünya mikrofon kaydında belirgin bir zemin uğultusu var.

**En iyi gürültü bastırıcılar (en düşük noise floor):**
1. RNNoise: −77.6 dB (60 dB iyileşme)
2. Demucs dns64: −77.3 dB (60 dB iyileşme)
3. Demucs dns48: −71.4 dB (54 dB iyileşme)
4. DeepFilterNet: −65.7 dB (48 dB iyileşme)

**Zayıf bastırıcılar:**
- Spectral Subtraction: −34.5 dB (sadece 17 dB iyileşme; baseline olarak beklendiği gibi)
- MetricGAN+: −44.0 dB (orta)

### 5.2 Kritik Bulgu 1 — RNNoise Paradoksu

RNNoise sessiz anlarda mükemmel temizleme yapıyor (noise floor −77 dB) ama **HF Ratio = 0.0185**, tüm modellerin **en yükseği** ve orijinalin 5 katı. Bu şu anlama gelir: **Konuşma anlarında yüksek frekanslı artefaktlar (cızırtı/metalik ton) ekliyor.** Sessizlikte temizlemesi etkileyici ama konuşma üzerine müdahalesi bozucu. Bu bulgu, sübjektif "RNNoise hoşuma gitmedi" gözlemini sayısal olarak açıklıyor.

### 5.3 Kritik Bulgu 2 — MetricGAN+ Aşırı Agresif

MetricGAN+ üç ayrı metrikte uç değerler gösteriyor:
- RMS = −19.1 dB (diğerlerinin 2 katı sessiz). Model **genel ses seviyesini bastırıyor**.
- Centroid = 1022 Hz (diğerleri 1400–1900 arası). Ses **boğuk**.
- HF Ratio = 0.0002 (neredeyse sıfır). Tüm yüksek frekansları silmiş.

Bu profil, PESQ optimizasyonunun bilinen bir yan etkisidir: model "PESQ skoru yüksek çıktı" öğrenmek için sesin enerjisini ve tınısını kırpmayı seçmiş. Gürültü bastırma pahasına **konuşma doğallığı** kaybedilmiş.

### 5.4 Kritik Bulgu 3 — Demucs DNS64 vs Master64 Anomalisi

Aynı mimari, aynı parametre sayısı (33.5M), aynı boyut (128 MB):
- dns64: RTF = **0.301**, Noise Floor = −77.3 dB
- master64: RTF = **0.080**, Noise Floor = −48.6 dB

master64 **3.7 kat hızlı** ama gürültü bastırma konusunda **dns64'ten belirgin zayıf**. Bu fark muhtemelen master64'ün eğitildiği dataset farkından kaynaklanıyor (DNS + Valentini karışık eğitim); Valentini verisinin "daha yumuşak" bastırma eğitmesi söz konusu olabilir. Hız farkı ise tutarlı değil — iki model de aynı mimaride; ölçümde tek seferlik bir varyans olmuş olabilir, daha fazla çalıştırmayla doğrulanmalı.

### 5.5 Hız Analizi

**Mobil real-time güvenli bölge (PC'de RTF < 0.2):**
- metricgan_plus: 0.018 — ama kalite kabul edilemez
- spectral_subtraction: 0.004 — ama kalite baseline seviyesinde
- deepfilternet: 0.043 — **kalite/hız dengesi en iyi**
- demucs_master64: 0.080
- demucs_dns48: 0.104
- rnnoise: 0.162 — ama Python wrapper overhead'i yüzünden gerçek RNNoise'ın C versiyonundan çok daha yavaş. Android'de native entegrasyon ile çok daha iyi olur.

**Sınırda:**
- demucs_dns64: 0.301 — PC'de 3× gerçek zamanlı ama mobilde zor.

### 5.6 Boyut Analizi

Android APK boyut kısıtları:
- **İdeal (<10 MB):** spectral_subtraction, rnnoise (native ~85 KB), deepfilternet (8 MB), metricgan_plus (7 MB)
- **Sınırda (50–100 MB):** demucs_dns48 (72 MB)
- **Zor (>100 MB):** demucs_dns64, demucs_master64 (128 MB)

---

## 6. Mobil Hedef İçin Model Sıralaması

Aşağıdaki sıralama bu deneyin **tek dosya gözleminden** çıkarılmıştır; sentetik veri üstünde SNR farklılıklarıyla tekrar doğrulanmalıdır. Yine de ilk aday şeması şöyle:

### Birincil Aday: DeepFilterNet
**Gerekçe:** Hız (RTF 0.043), boyut (8 MB), kalite (Noise Floor −66 dB, dengeli frekans dağılımı, HF artefakt yok) üçünde de iyi denge sunuyor. Benchmark'ta belirgin bir zayıf noktası yok.

### İkincil Aday: RNNoise (Android'de native C entegrasyonu ile)
**Gerekçe:** Gerçek boyutu ~85 KB, yani en mobil dostu seçenek. PC Python wrapper'ındaki yavaşlık mobilde C olarak kullanıldığında ortadan kalkacaktır. Ancak HF Ratio problemi bir endişedir; konuşma üzerinde cızırtı riski nedeniyle DeepFilterNet'e ikinci sırada.

### Üçüncül Aday: Demucs dns48
**Gerekçe:** Kalite en iyilerinden (Noise Floor −71 dB, temiz spektrum) ve RTF 0.1 ile mobilde marjinal olarak geçer. Ama 72 MB boyut sınırda; kuantize edilmiş (INT8) sürümle 20 MB civarına inebilir.

### Elenenler
- **spectral_subtraction:** Sadece baseline, mobil üretim için yeterli kalite değil.
- **demucs_dns64 / demucs_master64:** 128 MB boyut ve dns64'te RTF sorunu.
- **metricgan_plus:** Boyut ve hız iyi ama ses kalitesi bozuk (aşırı agresif).

---

## 7. Önemli Kısıtlar ve Uyarılar

1. **Tek örnekli deney.** Sonuçlar 10 saniyelik tek bir mikrofon kaydına dayanıyor. Farklı gürültü tiplerinde ve SNR seviyelerinde sıralama değişebilir.

2. **Objektif kalite metriği yok.** PESQ/STOI/SI-SDR gibi referanslı metrikler hesaplanmadı; bu metrikler sentetik (temiz + gürültü) verisinde bir sonraki aşamada eklenecek.

3. **İlk-çağrı etkisi.** PyTorch modellerinin ilk forward-pass'i genelde sonrakilerden yavaştır (internal cache'ler doluyor). Benchmark'ta warmup yapılmadı; gerçek real-time kullanımda sürekli çalışan modelin performansı biraz daha iyi olacaktır.

4. **CPU-only ölçüm.** GPU hesaba katılmadı; mobilde de GPU kullanımı farklı bir hikaye, NNAPI/CoreML/DSP seçenekleri ayrıca değerlendirilmeli.

5. **RAM ölçüm hassasiyeti.** `psutil` ile "öncesi/sonrası" farkı alındı; peak bellek değil. Bazı satırlardaki negatif değerler garbage collection artefaktıdır.

## 8. Sıradaki Adımlar

1. **Objektif kalite ölçümü** — PESQ, STOI, SI-SDR; sentetik karışım verisiyle.
2. **Farklı SNR seviyelerinde doğrulama** — −5, 0, 5, 10, 15 dB.
3. **Farklı gürültü tipleri** — DEMAND, MUSAN ile çoklu senaryo.
4. **Bu deneyin birkaç kez tekrarı** — ölçüm gürültüsünü anlamak için.
5. **Kuantizasyon denemesi** — Demucs dns48'in INT8 versiyonu mobilde 20 MB altına inebilir mi?
6. **Android tarafı ölçümü** — seçilen aday(lar)ın ONNX/TFLite'a dönüşümü ve gerçek cihazda RTF testi.

---

*Rapor otomatik olarak `benchmark_all.py` çıktılarından ve ses dosyalarının sinyal analizinden üretilmiştir.*
