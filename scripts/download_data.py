"""Dataset indirme ve hazırlama scripti.

Senaryo S için gerekli üç kaynak:
  1. VCTK (İngilizce konuşma) — 10 konuşmacı × 1 cümle = 10 wav  -> data/clean/en/
  2. Common Voice Turkish (Türkçe konuşma) — 10 wav             -> data/clean/tr/
  3. DEMAND (gürültü) — 7 sahne, her sahnenin ch01.wav'ı        -> data/noise/{SCENE}/

Tüm sesleri 16 kHz mono float32'ye normalize ederek kaydeder. Idempotent: ikinci
çağrıda mevcut dosyaları atlar. Bir dataset inmezse diğerlerine devam eder.

Kullanım:
    python -m scripts.download_data
    python -m scripts.download_data --only vctk        # sadece bir kaynak
    python -m scripts.download_data --only demand
    python -m scripts.download_data --force            # mevcut dosyaları yeniden indir

Notlar:
  - Common Voice indirmek için Hugging Face hesabı ve `huggingface-cli login`
    gerekebilir (Mozilla dataset'in lisansını kabul etmek zorunlu).
  - DEMAND Zenodo'dan indirilir, sahne başına ~80-200 MB zip (toplam ~1 GB).
  - VCTK indirme HuggingFace `CSTR-Edinburgh/vctk` üzerinden veya datashare
    üzerinden olabilir; biz HF API kullanıyoruz (auth gerektirmez genelde).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

from audio_io.file_io import load_audio, save_audio


# ---------------------------------------------------------------------------
# Klasör sabitleri
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data")
CLEAN_EN_DIR = DATA_ROOT / "clean" / "en"
CLEAN_TR_DIR = DATA_ROOT / "clean" / "tr"
NOISE_DIR = DATA_ROOT / "noise"

# ---------------------------------------------------------------------------
# VCTK seçimi: 5 erkek + 5 kadın, çeşitli aksanlar
# Konuşmacı ID'leri ve cinsiyetleri VCTK speaker-info dosyasından.
# ---------------------------------------------------------------------------

VCTK_SPEAKERS: list[tuple[str, str]] = [
    # (speaker_id, gender)
    ("p225", "F"),  # English (Southern England)
    ("p226", "M"),  # English (Surrey)
    ("p227", "M"),  # English (Cumbria)
    ("p228", "F"),  # English (Southern England)
    ("p229", "F"),  # English (Southern England)
    ("p230", "F"),  # English (Stockton-on-tees)
    ("p231", "F"),  # English (Southern England)
    ("p232", "M"),  # English (Southern England)
    ("p233", "F"),  # English (Staffordshire)
    ("p234", "M"),  # Scottish (West Dumfries)
]

# DEMAND 7 sahne (zenodo records 1227121)
# Her sahnenin URL'i: https://zenodo.org/record/1227121/files/{SCENE}.zip
DEMAND_SCENES: list[str] = [
    "TBUS",      # Otobüs içi
    "TCAR",      # Araç içi
    "TMETRO",   # Metro
    "SCAFE",    # Kafe (kalabalık konuşma)
    "SPSQUARE", # Halk meydanı (kalabalık konuşma)
    "OOFFICE",  # Ofis (sabit fan/uğultu)
    "NPARK",    # Park (rüzgar / doğa)
]

DEMAND_BASE_URL = "https://zenodo.org/record/1227121/files"


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _ensure_dirs():
    CLEAN_EN_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_TR_DIR.mkdir(parents=True, exist_ok=True)
    NOISE_DIR.mkdir(parents=True, exist_ok=True)


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _normalize_and_save(src_audio: np.ndarray, src_sr: int, dest: Path) -> None:
    """Audio'yu 16 kHz mono float32'ye normalize edip dest'e yaz.

    src_audio mono değilse ortalama alınır. load_audio zaten yapıyor ama
    bazı code path'leri (Common Voice raw bytes) bypass edebilir.
    """
    if src_audio.ndim > 1:
        src_audio = src_audio.mean(axis=-1)
    src_audio = src_audio.astype(np.float32)

    # Yeniden örnekleme + normalizasyon load_audio ile değil çünkü zaten array elimizde.
    # Basit yol: dest'e write etmeden önce geçici bir array oluştur ve resample uygula.
    if src_sr != 16000:
        from scipy.signal import resample_poly
        src_audio = resample_poly(src_audio, up=16000, down=src_sr).astype(np.float32)

    dest.parent.mkdir(parents=True, exist_ok=True)
    save_audio(str(dest), src_audio, sr=16000)


def _retry_action(name: str, fn: Callable[[], None]) -> bool:
    """Eylemi çalıştır, başarılı/başarısız durumunu logla, hatayı yutmasın."""
    try:
        fn()
        return True
    except Exception as e:
        print(f"  [HATA] {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# VCTK
# ---------------------------------------------------------------------------

def download_vctk(force: bool = False) -> int:
    """10 VCTK konuşmacısının ilk cümlesini indir.

    HuggingFace 'CSTR-Edinburgh/vctk' (veya benzeri mirror) deneniyor.
    Auth gerekmez genellikle. Başarısız olursa kullanıcıya alternatif söyleniyor.

    Returns: başarıyla indirilen dosya sayısı.
    """
    print("\n=== VCTK (English clean speech) ===")

    # Çoğu dosya zaten varsa erken çık
    existing = [
        CLEAN_EN_DIR / f"spk{spk}_utt001.wav" for spk, _ in VCTK_SPEAKERS
    ]
    if not force and all(_exists(p) for p in existing):
        print(f"  Tüm {len(existing)} dosya zaten var, atlanıyor.")
        return len(existing)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  [HATA] 'datasets' paketi kurulu değil. Kurmak için: pip install datasets")
        return 0

    # HF'de VCTK için yaygın mirror'lar
    candidate_repos = [
        "CSTR-Edinburgh/vctk",
        "vctk",
    ]

    ds = None
    for repo in candidate_repos:
        try:
            print(f"  HF'de '{repo}' deneniyor (streaming)...")
            ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            print(f"    [{repo}] başarısız: {type(e).__name__}: {str(e)[:120]}")

    if ds is None:
        print("  VCTK indirilemedi. Manuel: https://datashare.ed.ac.uk/handle/10283/3443")
        return 0

    wanted_speakers = {spk: gender for spk, gender in VCTK_SPEAKERS}
    found: dict[str, bool] = {spk: False for spk in wanted_speakers}
    count = 0

    for sample in ds:
        # HF VCTK sample alanları: 'audio' (dict: array, sampling_rate, path), 'speaker_id', ...
        spk_id = sample.get("speaker_id") or sample.get("speaker") or ""
        # Bazı mirror'larda speaker_id 'p225' yerine '225' olabilir
        spk_norm = spk_id if spk_id.startswith("p") else f"p{spk_id}"
        if spk_norm not in wanted_speakers or found[spk_norm]:
            continue

        audio = sample.get("audio")
        if not audio or "array" not in audio:
            continue

        dest = CLEAN_EN_DIR / f"spk{spk_norm}_utt001.wav"
        if not force and _exists(dest):
            found[spk_norm] = True
            continue

        ok = _retry_action(
            f"VCTK {spk_norm}",
            lambda a=audio, d=dest: _normalize_and_save(
                np.asarray(a["array"], dtype=np.float32),
                int(a["sampling_rate"]),
                d,
            ),
        )
        if ok:
            found[spk_norm] = True
            count += 1
            print(f"  + {dest.name}")

        if all(found.values()):
            break

    missing = [s for s, ok in found.items() if not ok]
    if missing:
        print(f"  Eksik konuşmacılar: {missing}")
    print(f"  VCTK: {count} yeni dosya indirildi, toplam {sum(found.values())}/{len(wanted_speakers)}")
    return count


# ---------------------------------------------------------------------------
# Common Voice Turkish
# ---------------------------------------------------------------------------

def download_cv_tr(force: bool = False) -> int:
    """Common Voice Turkish'ten kalite filtresine takılan 10 dosya indir.

    Filtre: up_votes >= 2, down_votes == 0. Çeşitli konuşmacılar.

    NOT: Common Voice gated dataset. `huggingface-cli login` gerekir + Mozilla
    sayfasında dataset'i kabul etmek gerekir.
    """
    print("\n=== Common Voice Turkish ===")

    existing = sorted(CLEAN_TR_DIR.glob("spk*_utt*.wav"))
    if not force and len(existing) >= 10:
        print(f"  Tüm 10 dosya zaten var ({len(existing)} bulundu), atlanıyor.")
        return 10

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  [HATA] 'datasets' paketi kurulu değil. pip install datasets")
        return 0

    candidate_repos = [
        ("mozilla-foundation/common_voice_17_0", "tr"),
        ("mozilla-foundation/common_voice_16_1", "tr"),
        ("mozilla-foundation/common_voice_13_0", "tr"),
    ]

    ds = None
    for repo, lang in candidate_repos:
        try:
            print(f"  HF'de '{repo}' (tr) deneniyor (streaming)...")
            ds = load_dataset(repo, lang, split="validated", streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            err = str(e)[:200]
            print(f"    [{repo}] başarısız: {type(e).__name__}: {err}")
            if "gated" in err.lower() or "401" in err or "403" in err:
                print(
                    "    Common Voice gated dataset. Önce: "
                    "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 "
                    "sayfasında accept + `huggingface-cli login`."
                )

    if ds is None:
        print("  Common Voice TR indirilemedi.")
        return 0

    seen_clients: set[str] = set()
    count = 0
    for sample in ds:
        if count >= 10:
            break
        up = int(sample.get("up_votes", 0))
        down = int(sample.get("down_votes", 0))
        if up < 2 or down > 0:
            continue
        client = sample.get("client_id", "")
        if client in seen_clients:
            continue  # konuşmacı çeşitliliği
        audio = sample.get("audio")
        if not audio or "array" not in audio:
            continue

        spk_short = (client[:8] if client else f"spk{count:02d}")
        dest = CLEAN_TR_DIR / f"spk{spk_short}_utt{count+1:03d}.wav"
        if not force and _exists(dest):
            seen_clients.add(client)
            count += 1
            continue

        ok = _retry_action(
            f"CV-TR sample {count+1}",
            lambda a=audio, d=dest: _normalize_and_save(
                np.asarray(a["array"], dtype=np.float32),
                int(a["sampling_rate"]),
                d,
            ),
        )
        if ok:
            seen_clients.add(client)
            count += 1
            print(f"  + {dest.name}")

    print(f"  CV-TR: {count}/10 dosya indirildi.")
    return count


# ---------------------------------------------------------------------------
# DEMAND
# ---------------------------------------------------------------------------

def download_demand(force: bool = False) -> int:
    """DEMAND'dan 7 sahnenin ch01.wav'larını indir.

    Zenodo zip dosyaları üzerinden. Sahne başına ~80-200 MB zip; sadece ch01.wav
    çıkarılır, zip diske yazılmaz (memory-only unzip).
    """
    print("\n=== DEMAND (noise) ===")

    try:
        import requests  # type: ignore
    except ImportError:
        print("  [HATA] 'requests' paketi kurulu değil. pip install requests")
        return 0

    count = 0
    for scene in DEMAND_SCENES:
        dest_dir = NOISE_DIR / scene
        dest = dest_dir / "ch01.wav"
        if not force and _exists(dest):
            print(f"  {scene}/ch01.wav zaten var, atlanıyor.")
            continue

        url = f"{DEMAND_BASE_URL}/{scene}.zip"
        print(f"  {scene} indiriliyor: {url}")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            zip_bytes = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_bytes) as zf:
                # DEMAND zip'inde dosya yolu: {SCENE}/ch01.wav (genelde)
                # Bazı sürümlerde sadece ch01.wav olabilir, esnek arayalım.
                candidates = [n for n in zf.namelist() if n.endswith("ch01.wav")]
                if not candidates:
                    print(f"  [HATA] {scene}.zip içinde ch01.wav bulunamadı: {zf.namelist()[:5]}")
                    continue
                inner = candidates[0]
                with zf.open(inner) as f:
                    # Geçici dosyaya yaz (load_audio dosya yolu istiyor)
                    tmp = dest_dir / "_tmp_ch01.wav"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    with open(tmp, "wb") as out:
                        out.write(f.read())
                # 16 kHz mono normalize
                audio, _ = load_audio(str(tmp))
                save_audio(str(dest), audio, sr=16000)
                tmp.unlink(missing_ok=True)
                count += 1
                print(f"  + {scene}/ch01.wav ({len(audio) / 16000:.1f} s)")
        except Exception as e:
            print(f"  [HATA] {scene}: {type(e).__name__}: {str(e)[:160]}")

    print(f"  DEMAND: {count}/{len(DEMAND_SCENES)} sahne indirildi.")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCTK + Common Voice TR + DEMAND indirme.")
    p.add_argument(
        "--only",
        choices=["vctk", "cv_tr", "demand"],
        default=None,
        help="Sadece belirtilen dataseti indir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Mevcut dosyaları yeniden indir.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_dirs()

    total = 0
    if args.only in (None, "vctk"):
        total += download_vctk(force=args.force)
    if args.only in (None, "cv_tr"):
        total += download_cv_tr(force=args.force)
    if args.only in (None, "demand"):
        total += download_demand(force=args.force)

    print(f"\n=== ÖZET ===")
    print(f"  data/clean/en: {len(list(CLEAN_EN_DIR.glob('*.wav')))} wav")
    print(f"  data/clean/tr: {len(list(CLEAN_TR_DIR.glob('*.wav')))} wav")
    print(f"  data/noise/: {len(list(NOISE_DIR.glob('*/ch01.wav')))} sahne")
    print(f"  Bu çalıştırmada eklenen: {total} dosya")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
