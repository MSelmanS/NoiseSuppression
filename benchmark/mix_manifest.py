"""Pre-built mix manifest okuma/yazma yardımcıları.

Manifest dosyası: `input_data/{profile}/manifest.csv`
Sütunlar (sırayla, sabit):
  idx, lang, clean_id, clean_path, noise_scene, noise_path,
  noise_offset_sec, target_snr_db, achieved_snr_db, duration_sec,
  mix_filename, source

source = "generated" veya "copied_from:{profile}".
mix_filename sadece basename — tam yol için manifest klasörü prepend edilir.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict, fields


# Manifest CSV başlık sırası — DEĞİŞMESİN, geriye uyumluluk için sabit.
MANIFEST_FIELDS: list[str] = [
    "idx",
    "lang",
    "clean_id",
    "clean_path",
    "noise_scene",
    "noise_path",
    "noise_offset_sec",
    "target_snr_db",
    "achieved_snr_db",
    "duration_sec",
    "mix_filename",
    "source",
]


@dataclass
class MixEntry:
    idx: int
    lang: str
    clean_id: str
    clean_path: str
    noise_scene: str
    noise_path: str
    noise_offset_sec: float
    target_snr_db: float
    achieved_snr_db: float
    duration_sec: float
    mix_filename: str  # sadece basename
    source: str        # "generated" | "copied_from:{profile}"

    def to_row(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


def manifest_path_for(profile: str, root: str = "input_data") -> str:
    """`input_data/{profile}/manifest.csv` yolu."""
    return os.path.join(root, profile, "manifest.csv")


def mix_path_for(profile: str, mix_filename: str, root: str = "input_data") -> str:
    """`input_data/{profile}/{mix_filename}` yolu."""
    return os.path.join(root, profile, mix_filename)


def save_manifest(path: str, entries: list[MixEntry]) -> None:
    """Manifest'i UTF-8 CSV olarak yaz (mevcut dosyayı ezerse ezer)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # idx'e göre sırala — okunabilirlik için
    sorted_entries = sorted(entries, key=lambda e: e.idx)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for e in sorted_entries:
            writer.writerow(e.to_row())


def load_manifest(path: str) -> list[MixEntry]:
    """Manifest'i oku, MixEntry listesi döner. Dosya yoksa boş liste."""
    if not os.path.isfile(path):
        return []
    out: list[MixEntry] = []
    # Hangi alanlar dataclass'ta var
    valid_keys = {f.name for f in fields(MixEntry)}
    type_map: dict[str, type] = {
        "idx": int,
        "lang": str,
        "clean_id": str,
        "clean_path": str,
        "noise_scene": str,
        "noise_path": str,
        "noise_offset_sec": float,
        "target_snr_db": float,
        "achieved_snr_db": float,
        "duration_sec": float,
        "mix_filename": str,
        "source": str,
    }
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kwargs: dict[str, object] = {}
            for k, v in row.items():
                if k not in valid_keys:
                    continue
                kwargs[k] = type_map[k](v) if v != "" else ("" if type_map[k] is str else 0)
            try:
                out.append(MixEntry(**kwargs))  # type: ignore[arg-type]
            except TypeError:
                # eksik kolon vs.; o satırı atla
                continue
    return out


def verify_manifest(path: str) -> tuple[list[MixEntry], list[str]]:
    """Manifest'i oku ve her satırın işaret ettiği wav diskte var mı kontrol et.

    Returns: (found_entries, missing_filenames)
    """
    entries = load_manifest(path)
    if not entries:
        return [], []
    base = os.path.dirname(path)
    missing: list[str] = []
    for e in entries:
        full = os.path.join(base, e.mix_filename)
        if not os.path.isfile(full):
            missing.append(e.mix_filename)
    return entries, missing


def find_existing_mix(filename: str, root: str = "input_data") -> str | None:
    """Bu filename'i diğer profil klasörlerinde ara; ilk bulduğu tam yolu dön.

    Profil arası yeniden kullanım için. Bulunmazsa None.
    """
    if not os.path.isdir(root):
        return None
    for profile in sorted(os.listdir(root)):
        candidate = os.path.join(root, profile, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def snr_tag(snr: float) -> str:
    """SNR değerini iki haneli, sayısal sıralanabilir tag'e çevir.

    -5  -> "-05"
     0  -> "00"
     5  -> "05"
    10  -> "10"
    """
    n = int(round(snr))
    if n < 0:
        return f"-{abs(n):02d}"
    return f"{n:02d}"


def mix_filename(lang: str, clean_id: str, noise_scene: str, snr: float) -> str:
    """Standart mix dosya adı: `{lang}_{clean_id}__{noise_scene}__snr{XX}.wav`."""
    return f"{lang}_{clean_id}__{noise_scene}__snr{snr_tag(snr)}.wav"
