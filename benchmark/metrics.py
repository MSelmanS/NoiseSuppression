"""Benchmark ölçüm yardımcıları: zaman, RAM (anlık ve peak), model boyutu,
ve objektif kalite metrikleri (SI-SDR, STOI, PESQ).
"""

import os
import time
import threading
import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Zaman ve RAM
# ---------------------------------------------------------------------------

def current_ram_mb():
    """Mevcut process'in RAM kullanımı (MB)."""
    process = psutil.Process(os.getpid())
    # rss = Resident Set Size: process'in fiziksel bellekteki boyutu
    return process.memory_info().rss / (1024 * 1024)


def time_it(fn, *args, **kwargs):
    """Bir fonksiyonu çalıştır, süresini ve sonucunu döndür."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class PeakRSSTracker:
    """Arka plan thread'inde process RSS'ini örnekler, peak değeri tutar.

    Tek seferlik anlık ölçüm yetersiz: model.load() ve process() arasında RAM
    pik yapıp düşebilir, sonradan bakınca düşük görünür. Bu sınıf with bloğu
    süresince ~interval_s aralıkla örnek alıp en yüksek değeri saklar.

    Kullanım:
        with PeakRSSTracker() as t:
            model.load()
            model.process(audio)
        print(t.peak_mb)
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._process = psutil.Process(os.getpid())
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self):
        # Thread başlamadan önceki anlık değeri de baseline olarak al
        while not self._stop_event.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except psutil.Error:
                # Process bir nedenle erişilemez olursa thread sessizce sonlansın
                break
            # Event.wait timeout'u, sleep'e göre stop sinyaline daha hızlı tepki verir
            self._stop_event.wait(self.interval_s)

    def __enter__(self) -> "PeakRSSTracker":
        # Başlangıç değerini ölç ki çok kısa with blokları bile bir peak versin
        self._peak_bytes = self._process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Çıkıştan hemen önce de son bir örnek al
        try:
            rss = self._process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
        except psutil.Error:
            pass
        return False

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Model boyutu / parametre sayısı
# ---------------------------------------------------------------------------

def model_size_mb(model):
    """PyTorch modelinin toplam parametre boyutunu MB olarak hesapla.
    PyTorch olmayan modellerde 0 döner."""
    try:
        if not hasattr(model, "parameters"):
            return 0.0
        # Her parametrenin element sayısı * byte cinsinden boyutu toplanır
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def param_count(model):
    """PyTorch modelinin toplam parametre sayısı. PyTorch değilse 0."""
    try:
        if not hasattr(model, "parameters"):
            return 0
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Objektif kalite metrikleri (referans gerektirir)
# ---------------------------------------------------------------------------

def _align(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """İki sinyali aynı uzunluğa kırp. Resample/STFT sonrası 1-2 sample fark olabilir."""
    n = min(len(reference), len(estimate))
    return reference[:n].astype(np.float32), estimate[:n].astype(np.float32)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB).

    Formül:
        proj    = <est, ref> / ||ref||^2 * ref
        e_noise = est - proj
        SI-SDR  = 10 * log10(||proj||^2 / ||e_noise||^2)

    Yüksek = iyi. Tipik aralık -10..+30 dB.
    """
    ref, est = _align(reference, estimate)

    # Zero-mean (yaygın konvansiyon; DC bias SI-SDR'ı bozmasın)
    ref = ref - ref.mean()
    est = est - est.mean()

    ref_energy = np.dot(ref, ref) + 1e-12
    proj_scale = np.dot(est, ref) / ref_energy
    proj = proj_scale * ref
    noise = est - proj

    proj_energy = np.dot(proj, proj) + 1e-12
    noise_energy = np.dot(noise, noise) + 1e-12

    return float(10.0 * np.log10(proj_energy / noise_energy))


def stoi_score(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> float:
    """pystoi.stoi wrapper. Aralık 0..1, yüksek = anlaşılır.

    pystoi yoksa veya hata olursa NaN döner.
    """
    try:
        from pystoi import stoi
    except ImportError:
        return float("nan")

    ref, est = _align(reference, estimate)
    try:
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return float("nan")


def pesq_score(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> float:
    """pesq paketi wrapper. Aralık ~1..4.5, yüksek = iyi.

    sr == 16000 -> wideband ('wb'), sr == 8000 -> narrowband ('nb').
    PESQ bazı dosyalarda hata fırlatır (sessiz veya kısa sinyaller); o durumda NaN.
    pesq kurulu değilse de NaN.
    """
    try:
        from pesq import pesq
    except ImportError:
        return float("nan")

    if sr == 16000:
        mode = "wb"
    elif sr == 8000:
        mode = "nb"
    else:
        # PESQ sadece 8k/16k destekler; başka SR ise NaN
        return float("nan")

    ref, est = _align(reference, estimate)
    try:
        return float(pesq(sr, ref, est, mode))
    except Exception:
        return float("nan")
