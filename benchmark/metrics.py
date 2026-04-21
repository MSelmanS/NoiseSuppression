"""Benchmark ölçüm yardımcıları: zaman, RAM, model boyutu."""

import os
import time
import psutil


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


def model_size_mb(model):
    """PyTorch modelinin toplam parametre boyutunu MB olarak hesapla.
    PyTorch olmayan modellerde 0 döner."""
    try:
        import torch
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
