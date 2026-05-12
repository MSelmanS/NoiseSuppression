"""Tek-dosya HTML rapor üreticisi.

Bench sonuç klasörünü (results_raw.csv + samples/ + plot_*.png + anomalies.csv)
okur, sekiz bölümlü kendi içinde tam HTML çıktısı üretir. Tüm ses dosyaları ve
PNG'ler base64 gömülü.

Bölümler (sırayla):
  1. Başlık & meta — deney tarihi, profil, dataset, model listesi, hipotez metni
  2. Liderlik tablosu — her metrik için model sıralaması (mean değerlerle)
  3. Hipotez testi — H1-H4 (benchmark.hypothesis modülünden)
  4. Sahne × SNR ısı haritaları
  5. Detay tablosu — model × sahne × SNR ortalama (per-SNR XLSX'in HTML eşleniği)
  6. Dinleme galerisi — samples klasöründeki wav + spektrogram
  7. Anomaliler — anomalies.csv tablo halinde
  8. Grafikler — plot_*.png'lerin embedded versiyonu
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from html import escape
from typing import Any

import numpy as np
import pandas as pd

from audio_io.file_io import load_audio
from benchmark.spectrogram import make_spectrogram_b64, _safe_db_spec, compute_global_vminmax
from benchmark.hypothesis import run_all as run_hypotheses


STATUS_COLOR = {
    "VERIFIED": "#16a34a",      # yeşil
    "PARTIAL": "#eab308",        # sarı
    "REJECTED": "#dc2626",       # kırmızı
    "INSUFFICIENT_DATA": "#6b7280",  # gri
}


# ---------------------------------------------------------------------------
# HTML şablon parçacıkları
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 1.5rem auto; padding: 0 1rem; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { border-bottom: 2px solid #888; padding-bottom: 0.3rem; margin-top: 2rem; }
h3 { margin-top: 1.2rem; }
.meta { color: #666; font-size: 0.9rem; }
.hypo-card { padding: 0.6rem 0.9rem; border-left: 4px solid #888; margin: 0.6rem 0;
             background: rgba(128,128,128,0.05); }
.status-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 0.3rem;
                color: white; font-weight: bold; font-size: 0.85rem; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #aaa; padding: 0.3rem 0.6rem; text-align: right; }
th { background: rgba(128,128,128,0.15); }
td:first-child, th:first-child { text-align: left; }
.heatmap-cell { font-weight: bold; }
.gallery-row { display: flex; gap: 0.8rem; margin: 0.5rem 0; flex-wrap: wrap; }
.gallery-cell { flex: 1; min-width: 260px; border: 1px solid #aaa; padding: 0.4rem; }
.gallery-cell img { width: 100%; height: auto; }
.gallery-cell audio { width: 100%; }
.anomaly-high { background: rgba(220,38,38,0.15); }
.anomaly-medium { background: rgba(234,179,8,0.15); }
.anomaly-low { background: rgba(96,165,250,0.10); }
.toc { font-size: 0.95rem; }
.toc a { text-decoration: none; }
.collapsible { border: 1px solid #888; padding: 0.4rem 0.8rem; margin: 0.4rem 0;
               border-radius: 0.3rem; }
.metric-num { font-variant-numeric: tabular-nums; }
""".strip()


# ---------------------------------------------------------------------------
# Yardımcı renkler & embedleme
# ---------------------------------------------------------------------------

def _heatmap_color(val: float, vmin: float, vmax: float, higher_better: bool = True) -> str:
    """[vmin, vmax] aralığında değer için yeşil-sarı-kırmızı renk."""
    if not np.isfinite(val) or vmin == vmax:
        return "transparent"
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if not higher_better:
        t = 1.0 - t
    # Yeşil (0,170,80) -> Sarı (240,200,40) -> Kırmızı (220,40,40)
    if t < 0.5:
        # red -> yellow
        u = t * 2
        r, g, b = (220 + (240 - 220) * u, 40 + (200 - 40) * u, 40 + (40 - 40) * u)
    else:
        u = (t - 0.5) * 2
        r, g, b = (240 + (0 - 240) * u, 200 + (170 - 200) * u, 40 + (80 - 40) * u)
    return f"rgba({int(r)},{int(g)},{int(b)},0.35)"


def _wav_to_b64(path: str) -> str:
    """Bir .wav dosyasını base64 string olarak oku (audio tag'inde kullanılır)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _png_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Bölüm üreticileri
# ---------------------------------------------------------------------------

def _section_header_meta(out_dir: str, df: pd.DataFrame, profile: str | None) -> str:
    timestamp = os.path.basename(out_dir).replace("bench_synthetic_", "")
    models = sorted(df["model"].unique().tolist()) if "model" in df.columns else []
    scenes = sorted(df["scene"].unique().tolist()) if "scene" in df.columns else []
    snrs = sorted(df["snr_db"].unique().tolist()) if "snr_db" in df.columns else []
    return f"""
<h1>Denoiser Benchmark Raporu</h1>
<p class="meta">
  Çalıştırma: <b>{escape(timestamp)}</b> &middot;
  Profil: <b>{escape(profile or '(yok)')}</b> &middot;
  Toplam ölçüm: <b>{len(df)}</b>
</p>
<p class="meta">Modeller ({len(models)}): {escape(', '.join(models))}</p>
<p class="meta">Sahneler ({len(scenes)}): {escape(', '.join(map(str, scenes)))}</p>
<p class="meta">SNR seviyeleri (dB): {escape(', '.join(f'{s:g}' for s in snrs))}</p>

<div class="collapsible">
  <b>Hipotez (1.4):</b> Klasik/hafif yöntemler düşük SNR'da zorlanır;
  DL modelleri sabit gürültüde güçlüdür; MetricGAN+ bazı durumlarda aşırı
  agresiftir; kalabalık konuşma sahnelerinde tüm modeller benzer zorluk yaşar.
  Detay alt-iddialar (H1-H4) Bölüm 3'te otomatik test ediliyor.
</div>
"""


def _section_leaderboard(df: pd.DataFrame) -> str:
    if "model" not in df.columns:
        return ""
    metrics = [
        ("pesq", "PESQ", True),
        ("stoi", "STOI", True),
        ("si_sdr", "SI-SDR (dB)", True),
        ("rtf_mean", "RTF", False),
        ("peak_ram_mb", "Peak RAM (MB)", False),
    ]
    rows_html = []
    cols = [m for m, _, _ in metrics if m in df.columns]
    if not cols:
        return "<h2 id='leaderboard'>2. Liderlik Tablosu</h2><p>Metrik yok.</p>"
    agg = df.groupby("model")[cols].mean(numeric_only=True).round(3)
    headers = "".join(f"<th>{escape(lbl)}</th>" for _, lbl, _ in metrics if _ in df.columns or True)
    # Düzeltme: sadece df'de bulunan metriklerin başlıklarını al
    headers = "".join(f"<th>{escape(lbl)}</th>" for m, lbl, _ in metrics if m in cols)

    for model_name, row in agg.iterrows():
        cells = []
        for m, _, _ in metrics:
            if m not in cols:
                continue
            v = row[m]
            cells.append(f"<td class='metric-num'>{v:.3f}</td>" if pd.notna(v) else "<td>-</td>")
        rows_html.append(f"<tr><td>{escape(str(model_name))}</td>{''.join(cells)}</tr>")

    return f"""
<h2 id="leaderboard">2. Liderlik Tablosu</h2>
<p>Tüm SNR/sahne/pair üzerinden ortalama. Her metrikte en iyiyi mavi vurgulamak için ileride
   geliştirilebilir; şimdilik ham mean.</p>
<table>
  <tr><th>Model</th>{headers}</tr>
  {''.join(rows_html)}
</table>
"""


def _section_hypothesis(hypothesis_results: list[dict[str, Any]]) -> str:
    cards = []
    for h in hypothesis_results:
        status = h.get("status", "INSUFFICIENT_DATA")
        color = STATUS_COLOR.get(status, "#666")
        evidence_lines = []
        for k, v in (h.get("evidence") or {}).items():
            if isinstance(v, (list, dict)):
                evidence_lines.append(f"<li><b>{escape(str(k))}:</b> {escape(str(v))}</li>")
            else:
                evidence_lines.append(f"<li><b>{escape(str(k))}:</b> {escape(str(v))}</li>")
        cards.append(f"""
<div class="hypo-card" style="border-left-color: {color};">
  <h3>{escape(h.get('id', ''))} — {escape(h.get('title', '(başlık yok)'))} &nbsp;
      <span class="status-badge" style="background:{color};">{escape(status)}</span></h3>
  <ul>{''.join(evidence_lines) or '<li>Veri yok</li>'}</ul>
</div>
""")
    return f"""
<h2 id="hypothesis">3. Hipotez Testi (H1-H4)</h2>
{''.join(cards)}
"""


def _section_heatmaps(df: pd.DataFrame, metric: str = "pesq", label: str = "PESQ") -> str:
    if metric not in df.columns or "scene" not in df.columns or "snr_db" not in df.columns:
        return ""
    pivot = df.pivot_table(index="scene", columns="snr_db", values=metric, aggfunc="mean")
    if pivot.empty:
        return ""
    vmin = float(pivot.min().min())
    vmax = float(pivot.max().max())
    snrs = list(pivot.columns)
    headers = "".join(f"<th>{snr:g} dB</th>" for snr in snrs)
    body = ""
    for scene, row in pivot.iterrows():
        cells = []
        for snr in snrs:
            v = row[snr]
            if pd.notna(v):
                color = _heatmap_color(v, vmin, vmax, higher_better=True)
                cells.append(
                    f'<td class="heatmap-cell metric-num" style="background:{color};">'
                    f'{v:.2f}</td>'
                )
            else:
                cells.append("<td>-</td>")
        body += f"<tr><td>{escape(str(scene))}</td>{''.join(cells)}</tr>"
    return f"""
<h2 id="heatmap">4. {escape(label)} Isı Haritası (Sahne × SNR)</h2>
<table>
  <tr><th>Scene</th>{headers}</tr>
  {body}
</table>
"""


def _section_detail_table(df: pd.DataFrame) -> str:
    """Model × Sahne × SNR ortalama PESQ tablosu."""
    if "pesq" not in df.columns or "scene" not in df.columns:
        return ""
    pivot = df.pivot_table(
        index=["model", "scene"],
        columns="snr_db",
        values="pesq",
        aggfunc="mean",
    ).round(3)
    if pivot.empty:
        return ""
    snrs = list(pivot.columns)
    headers = "".join(f"<th>{s:g} dB</th>" for s in snrs)
    body = ""
    for (model, scene), row in pivot.iterrows():
        cells = "".join(
            f'<td class="metric-num">{row[s]:.2f}</td>' if pd.notna(row[s]) else "<td>-</td>"
            for s in snrs
        )
        body += f"<tr><td>{escape(str(model))}</td><td>{escape(str(scene))}</td>{cells}</tr>"
    return f"""
<h2 id="detail">5. Detay Tablo — Model × Sahne × SNR (PESQ ortalama)</h2>
<table>
  <tr><th>Model</th><th>Scene</th>{headers}</tr>
  {body}
</table>
"""


def _section_gallery(out_dir: str, df: pd.DataFrame) -> str:
    """Dinleme galerisi. samples/_reference + samples/{model}/ klasörlerini okur."""
    samples_dir = os.path.join(out_dir, "samples")
    ref_dir = os.path.join(samples_dir, "_reference")
    if not os.path.isdir(samples_dir) or not os.path.isdir(ref_dir):
        return "<h2 id='gallery'>6. Dinleme Galerisi</h2><p>samples/ klasörü yok (--save-strategy=samples ile çalıştır).</p>"

    # Önce global vmin/vmax bul (tüm input audio'lardan)
    sample_audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    all_spectras: list[np.ndarray] = []

    noisy_files = sorted([f for f in os.listdir(ref_dir) if f.startswith("noisy_")])
    for f in noisy_files:
        path = os.path.join(ref_dir, f)
        try:
            audio, sr = load_audio(path)
            sample_audio_cache[path] = (audio, sr)
            all_spectras.append(_safe_db_spec(audio, sr))
        except Exception:
            pass

    # Modellerin çıktılarını da global skala için ekle
    model_dirs = [
        d for d in os.listdir(samples_dir)
        if d != "_reference" and os.path.isdir(os.path.join(samples_dir, d))
    ]
    for m in model_dirs:
        for f in os.listdir(os.path.join(samples_dir, m)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(samples_dir, m, f)
            try:
                audio, sr = load_audio(path)
                sample_audio_cache[path] = (audio, sr)
                all_spectras.append(_safe_db_spec(audio, sr))
            except Exception:
                pass

    vmin, vmax = compute_global_vminmax(all_spectras) if all_spectras else (-80.0, 0.0)

    # Her (scene, snr) için bir satır; satır içinde noisy + her model output
    # noisy_*.wav adlarından scene + snr çıkar
    items: dict[str, dict[str, str]] = {}  # key: "{scene}_snr{N}dB", value: {model -> wav_path, "_noisy" -> wav_path}
    for f in noisy_files:
        # noisy_SCAFE_snr-5dB.wav
        key = f.replace("noisy_", "").replace(".wav", "")
        items.setdefault(key, {})["_noisy"] = os.path.join(ref_dir, f)

    for m in model_dirs:
        m_dir = os.path.join(samples_dir, m)
        for f in os.listdir(m_dir):
            if not f.endswith(".wav"):
                continue
            key = f.replace(".wav", "")
            items.setdefault(key, {})[m] = os.path.join(m_dir, f)

    if not items:
        return "<h2 id='gallery'>6. Dinleme Galerisi</h2><p>Hiç örnek bulunamadı.</p>"

    blocks = []
    for key in sorted(items.keys()):
        cell_html = []
        # Önce noisy
        if "_noisy" in items[key]:
            path = items[key]["_noisy"]
            audio, sr = sample_audio_cache.get(path, load_audio(path))
            b64_audio = _wav_to_b64(path)
            b64_spec = make_spectrogram_b64(audio, sr, vmin=vmin, vmax=vmax, title="Noisy input")
            cell_html.append(f"""
<div class="gallery-cell">
  <b>Noisy (input)</b>
  <img src="data:image/png;base64,{b64_spec}" alt="noisy spec"/>
  <audio controls preload="none">
    <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav"/>
  </audio>
</div>""")
        # Sonra her model çıktısı
        for m in sorted(model_dirs):
            if m not in items[key]:
                continue
            path = items[key][m]
            audio, sr = sample_audio_cache.get(path, load_audio(path))
            b64_audio = _wav_to_b64(path)
            b64_spec = make_spectrogram_b64(audio, sr, vmin=vmin, vmax=vmax, title=m)
            # Bu örneğin metrikleri (df'den ara)
            metric_text = _lookup_metrics_for_sample(df, m, key)
            cell_html.append(f"""
<div class="gallery-cell">
  <b>{escape(m)}</b><br/>
  <small>{escape(metric_text)}</small>
  <img src="data:image/png;base64,{b64_spec}" alt="{escape(m)} spec"/>
  <audio controls preload="none">
    <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav"/>
  </audio>
</div>""")
        blocks.append(
            f'<h3>{escape(key)}</h3>'
            f'<div class="gallery-row">{"".join(cell_html)}</div>'
        )

    return f"""
<h2 id="gallery">6. Dinleme Galerisi ({len(items)} örnek)</h2>
{''.join(blocks)}
"""


def _lookup_metrics_for_sample(df: pd.DataFrame, model: str, sample_key: str) -> str:
    """Sample key (örn. 'SCAFE_snr-5dB') -> df satırında PESQ/STOI/SI-SDR."""
    if "scene" not in df.columns:
        return ""
    # Parse: SCAFE_snr-5dB
    try:
        scene, snr_part = sample_key.rsplit("_snr", 1)
        snr_val = float(snr_part.replace("dB", ""))
    except ValueError:
        return ""
    matches = df[
        (df["model"] == model) & (df["scene"] == scene) & (df["snr_db"].astype(float) == snr_val)
    ]
    if len(matches) == 0:
        return ""
    row = matches.iloc[0]
    parts = []
    for k, fmt in [("pesq", "PESQ={:.2f}"), ("stoi", "STOI={:.2f}"), ("si_sdr", "SI-SDR={:.1f}dB")]:
        if k in row and pd.notna(row[k]):
            parts.append(fmt.format(row[k]))
    return " | ".join(parts)


def _section_anomalies(out_dir: str) -> str:
    path = os.path.join(out_dir, "anomalies.csv")
    if not os.path.isfile(path):
        return "<h2 id='anomalies'>7. Anomaliler</h2><p>Anomali yakalanmadı.</p>"
    try:
        an = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return "<h2 id='anomalies'>7. Anomaliler</h2><p>anomalies.csv okunamadı.</p>"
    if len(an) == 0:
        return "<h2 id='anomalies'>7. Anomaliler</h2><p>Anomali yakalanmadı.</p>"

    cols = ["model", "scene", "snr_db", "type", "severity", "details"]
    cols = [c for c in cols if c in an.columns]
    rows = []
    for _, r in an.iterrows():
        sev = str(r.get("severity", "")).lower()
        css_cls = f"anomaly-{sev}" if sev in ("high", "medium", "low") else ""
        cells = "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols)
        rows.append(f'<tr class="{css_cls}">{cells}</tr>')

    headers = "".join(f"<th>{escape(c)}</th>" for c in cols)
    return f"""
<h2 id="anomalies">7. Anomaliler ({len(an)})</h2>
<table>
  <tr>{headers}</tr>
  {''.join(rows)}
</table>
"""


def _section_plots(out_dir: str) -> str:
    plots = [
        ("plot_pesq_vs_snr.png", "PESQ vs SNR"),
        ("plot_sisdr_vs_snr.png", "SI-SDR vs SNR"),
        ("plot_stoi_vs_snr.png", "STOI vs SNR"),
        ("plot_rtf_vs_snr.png", "RTF vs SNR"),
        ("plot_peakram_vs_snr.png", "Peak RAM vs SNR"),
    ]
    blocks = []
    for fname, title in plots:
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            continue
        b64 = _png_to_b64(path)
        blocks.append(
            f'<h3>{escape(title)}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{escape(title)}" '
            f'style="max-width:600px;display:block;"/>'
        )
    if not blocks:
        return ""
    return f"""
<h2 id="plots">8. Grafikler</h2>
{''.join(blocks)}
"""


def _toc() -> str:
    return """
<nav class="toc">
  <b>İçindekiler:</b>
  <a href="#leaderboard">2. Liderlik</a> &middot;
  <a href="#hypothesis">3. Hipotez</a> &middot;
  <a href="#heatmap">4. Heatmap</a> &middot;
  <a href="#detail">5. Detay</a> &middot;
  <a href="#gallery">6. Galeri</a> &middot;
  <a href="#anomalies">7. Anomali</a> &middot;
  <a href="#plots">8. Grafikler</a>
</nav>
"""


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------

def generate_html_report(
    out_dir: str,
    rows: list[dict],
    profile: str | None = None,
    dest_filename: str = "report.html",
) -> str:
    """Tek dosya HTML raporu üret. Dönüş: yazılan dosyanın tam yolu."""
    df = pd.DataFrame(rows)
    hyp = run_hypotheses(df)

    parts = [
        f"<!doctype html><html lang='tr'><head><meta charset='utf-8'>",
        f"<title>Denoiser Benchmark — {os.path.basename(out_dir)}</title>",
        f"<style>{_CSS}</style></head><body>",
        _section_header_meta(out_dir, df, profile),
        _toc(),
        _section_leaderboard(df),
        _section_hypothesis(hyp),
        _section_heatmaps(df, "pesq", "PESQ"),
        _section_detail_table(df),
        _section_gallery(out_dir, df),
        _section_anomalies(out_dir),
        _section_plots(out_dir),
        f"<hr><p class='meta'>Generated: {datetime.now().isoformat(timespec='seconds')}</p>",
        "</body></html>",
    ]
    html = "\n".join(parts)
    dest = os.path.join(out_dir, dest_filename)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(html)
    return dest
