"""Mel-spectrogram PNG üretici.

HTML raporda her dinleme örneği için (input, output) yan yana çizilir.
Tüm spektrogramlarda aynı renk skalası (vmin/vmax) kullanılır — modeller
arası gözle kıyas mümkün olsun.
"""

from __future__ import annotations

import base64
import io

import numpy as np

# matplotlib backend'i headless ortam için Agg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_db_spec(audio: np.ndarray, sr: int, n_mels: int = 128, hop: int = 256) -> np.ndarray:
    """Log-mel spektrogram (dB), librosa kuruluysa onunla, değilse fallback."""
    try:
        import librosa  # type: ignore
        S = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sr,
            n_mels=n_mels,
            hop_length=hop,
            n_fft=1024,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except ImportError:
        # Fallback: scipy STFT magnitudunda log
        from scipy.signal import stft
        _, _, Z = stft(audio.astype(np.float32), fs=sr, nperseg=1024, noverlap=1024 - hop)
        mag = np.abs(Z) ** 2
        S_db = 10.0 * np.log10(mag + 1e-12)
        return S_db


def make_spectrogram_b64(
    audio: np.ndarray,
    sr: int,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (4.0, 2.0),
    dpi: int = 80,
    title: str | None = None,
) -> str:
    """Mel-spectrogram PNG'ini base64 encoded string olarak döndür.

    HTML'de <img src="data:image/png;base64,{result}"> ile gömülür.
    """
    S_db = _safe_db_spec(audio, sr)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[0, len(audio) / sr, 0, sr // 2],
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Freq (Hz)", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.02)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_global_vminmax(spectrograms_db: list[np.ndarray]) -> tuple[float, float]:
    """Çoklu spektrogramı aynı renk skalasında çizmek için global vmin/vmax."""
    if not spectrograms_db:
        return -80.0, 0.0
    all_vals = np.concatenate([s.ravel() for s in spectrograms_db])
    return float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
