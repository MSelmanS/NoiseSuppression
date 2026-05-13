"""Pre-built mix üretici.

Profil parametrelerine göre tüm (clean, noise, snr) kombinasyonlarını
`input_data/{profile}/` altına yazar ve `manifest.csv` üretir. Mix'ler
diskte kalır; `bench_synthetic --use-prebuilt yes` ile aynı dosyalar
benchmark'ta kullanılabilir.

Kullanım:
    python -m scripts.build_mixes                    # interaktif profil sorar
    python -m scripts.build_mixes --profile s_smoke  # direkt s_smoke üret
    python -m scripts.build_mixes --profile s_quick --limit 20
    python -m scripts.build_mixes --profile s_quick --force

Profil arası yeniden kullanım: aynı (clean, noise, snr) kombinasyonu başka
bir profil klasöründe varsa diske yazılmaz, kopyalanır.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path

# Direct çalıştırma için proje kökünü path'e ekle (python scripts/build_mixes.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from audio_io.file_io import load_audio, save_audio
from benchmark.mixer import mix_at_snr, extend_to_min_duration
from benchmark.metrics import rms_db
from benchmark.mix_manifest import (
    MANIFEST_FIELDS,
    MixEntry,
    find_existing_mix,
    load_manifest,
    manifest_path_for,
    mix_filename as build_mix_filename,
    mix_path_for,
    save_manifest,
)
from scripts.profiles import PROFILES, get_profile


# Pre-build sırasında clean dosyalarını kısa olanları aynalama ile uzatmak için
# minimum saniye eşiği (bench_synthetic'teki ile aynı).
DEFAULT_MIN_CLEAN_DURATION = 3.5


def _ask_profile_interactive() -> str:
    """Profil adı CLI'da verilmezse konsoldan sor."""
    print("=" * 50)
    print("  build_mixes — Profil Seçimi")
    print("=" * 50)
    names = list(PROFILES.keys())
    for i, name in enumerate(names, start=1):
        desc = PROFILES[name].get("description", "")
        print(f"  [{i}] {name:<10}  {desc}")
    print()
    try:
        ans = input(f"Profil [1-{len(names)}, varsayılan 1]: ").strip()
    except EOFError:
        ans = ""
    try:
        idx = int(ans) - 1 if ans else 0
    except ValueError:
        idx = 0
    if idx < 0 or idx >= len(names):
        idx = 0
    return names[idx]


def _list_clean_by_lang(clean_dir: str) -> dict[str, list[str]]:
    """`{lang: [wav_paths]}` — `clean_dir/{lang}/*.wav` taranır."""
    out: dict[str, list[str]] = {}
    if not os.path.isdir(clean_dir):
        return out
    for entry in sorted(os.listdir(clean_dir)):
        sub = os.path.join(clean_dir, entry)
        if not os.path.isdir(sub):
            continue
        wavs = sorted(glob(os.path.join(sub, "*.wav")))
        if wavs:
            out[entry] = wavs
    return out


def _list_noise_scenes(noise_dir: str) -> dict[str, str]:
    """`{scene_name: wav_path}` — `noise_dir/{SCENE}/ch01.wav` taranır.

    Sadece scene/ch01.wav formatı kabul edilir; data/noise altındaki standalone
    .wav'lar (pink.wav, white.wav vs.) atlanır.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(noise_dir):
        return out
    for entry in sorted(os.listdir(noise_dir)):
        sub = os.path.join(noise_dir, entry)
        if not os.path.isdir(sub):
            continue
        wav = os.path.join(sub, "ch01.wav")
        if os.path.isfile(wav):
            out[entry] = wav
    return out


def _select_clean_files(
    clean_by_lang: dict[str, list[str]],
    max_pairs: int,
    seed: int = 42,
    min_duration: float = DEFAULT_MIN_CLEAN_DURATION,
) -> list[tuple[str, str]]:
    """Her dilden eşit pay alarak max_pairs dosya seç.

    Returns: [(lang, clean_path), ...] — deterministik (seed=42 ile shuffle).
    """
    langs = sorted(clean_by_lang.keys())
    if not langs:
        return []
    # Dil başına kota
    per_lang = max_pairs // len(langs)
    remainder = max_pairs - per_lang * len(langs)
    rng = np.random.default_rng(seed)

    selected: list[tuple[str, str]] = []
    for i, lang in enumerate(langs):
        wavs = sorted(clean_by_lang[lang])
        # Deterministik shuffle
        order = list(range(len(wavs)))
        rng.shuffle(order)
        wavs_shuffled = [wavs[i] for i in order]
        # İlk i diller remainder pay alır
        quota = per_lang + (1 if i < remainder else 0)
        for w in wavs_shuffled[:quota]:
            selected.append((lang, w))
    return selected


def _deterministic_noise_offset(
    clean_id: str,
    noise_scene: str,
    noise_duration_sec: float,
    clean_duration_sec: float,
    seed: int = 42,
) -> float:
    """Bir (clean, noise) çifti için reproducible noise offset (saniye)."""
    max_offset = max(0.0, noise_duration_sec - clean_duration_sec)
    if max_offset <= 0:
        return 0.0
    key = f"{seed}|{clean_id}|{noise_scene}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    # h modulo by integer seconds for stability; we then return float seconds
    granularity = 1000  # 1 ms resolution
    return (h % (int(max_offset * granularity) + 1)) / granularity


def _slice_noise(noise: np.ndarray, sr: int, offset_sec: float, target_len: int) -> np.ndarray:
    """noise'tan `offset_sec` saniyeden başlayarak `target_len` örnek al."""
    offset_samples = int(offset_sec * sr)
    end = offset_samples + target_len
    if end <= len(noise):
        return noise[offset_samples:end]
    # Yetmezse modulo ile sar (rare; only if noise shorter than clean which we already pad/trim downstream)
    out = np.zeros(target_len, dtype=noise.dtype)
    n = len(noise) - offset_samples
    if n > 0:
        out[:n] = noise[offset_samples:]
    # Geri kalanı baştan tekrar et
    pos = max(n, 0)
    while pos < target_len:
        chunk = min(len(noise), target_len - pos)
        out[pos:pos + chunk] = noise[:chunk]
        pos += chunk
    return out


def _measure_achieved_snr(clean: np.ndarray, mixed: np.ndarray) -> float:
    """mixed = clean + scaled_noise; achieved SNR (dB)."""
    residual = mixed - clean
    return rms_db(clean) - rms_db(residual)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profil parametrelerine göre tüm mix'leri diske yaz.",
    )
    p.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default=None,
        help="Profil adı. Verilmezse interaktif sorar.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Mevcut dosyaları yeniden üret (idempotent kontrolü atla).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Test için: sadece ilk N kombinasyonu üret.",
    )
    p.add_argument(
        "--input-root",
        default="input_data",
        help="Çıktı kök klasörü (varsayılan: input_data).",
    )
    p.add_argument(
        "--no-cross-reuse",
        action="store_true",
        help="Diğer profillerden kopyalamayı devre dışı bırak (her zaman üret).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Clean seçimi + noise offset için RNG tohumu.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile_name = args.profile or _ask_profile_interactive()

    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 2

    clean_dir = profile["clean_dir"]
    noise_dir = profile["noise_dir"]
    snrs = profile["snrs"]
    max_pairs = profile["max_pairs"]

    # 1. Dosyaları tara
    clean_by_lang = _list_clean_by_lang(clean_dir)
    noise_scenes = _list_noise_scenes(noise_dir)
    if not clean_by_lang:
        print(f"Hata: {clean_dir} altında dil alt klasörü (en/, tr/) bulunamadı.", file=sys.stderr)
        return 2
    if not noise_scenes:
        print(f"Hata: {noise_dir} altında sahne alt klasörü (SCAFE/ch01.wav vb.) bulunamadı.", file=sys.stderr)
        return 2

    selected_cleans = _select_clean_files(clean_by_lang, max_pairs, seed=args.seed)
    if not selected_cleans:
        print("Hata: hiç clean dosya seçilemedi.", file=sys.stderr)
        return 2

    total_combos = len(selected_cleans) * len(noise_scenes) * len(snrs)
    if args.limit:
        total_combos = min(total_combos, args.limit)

    print(f"Profil: {profile_name}")
    print(f"  Diller: {list(clean_by_lang.keys())}")
    print(f"  Seçilen clean: {len(selected_cleans)} ({max_pairs} talep edildi)")
    print(f"  Noise sahneleri: {list(noise_scenes.keys())} ({len(noise_scenes)} adet)")
    print(f"  SNR'lar: {snrs}")
    print(f"  Toplam kombinasyon: {len(selected_cleans)} × {len(noise_scenes)} × {len(snrs)} = "
          f"{len(selected_cleans) * len(noise_scenes) * len(snrs)}")
    if args.limit:
        print(f"  --limit {args.limit} uygulanacak -> {total_combos} kombinasyon")
    print()

    out_dir = os.path.join(args.input_root, profile_name)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = manifest_path_for(profile_name, root=args.input_root)

    # 2. Mevcut manifesti oku — idempotent kontrolü için
    existing_entries: list[MixEntry] = []
    existing_filenames: set[str] = set()
    if not args.force and os.path.isfile(manifest_path):
        existing_entries = load_manifest(manifest_path)
        existing_filenames = {
            e.mix_filename for e in existing_entries
            if os.path.isfile(os.path.join(out_dir, e.mix_filename))
        }
        print(f"Mevcut manifest: {len(existing_filenames)}/{len(existing_entries)} dosya doğrulandı.")

    # 3. Audio cache (clean ve noise dosyaları için disk-okuma masrafını azalt)
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _load_cached(path: str) -> tuple[np.ndarray, int]:
        if path not in audio_cache:
            audio_cache[path] = load_audio(path)
        return audio_cache[path]

    # 4. Kombinasyonları işle
    try:
        from tqdm import tqdm  # type: ignore
        progress = tqdm(total=total_combos, unit="mix")
    except ImportError:
        progress = None

    entries: list[MixEntry] = list(existing_entries) if not args.force else []
    # idx çakışmasın diye max idx'i takip et
    next_idx = (max((e.idx for e in entries), default=-1) + 1) if entries else 0

    stats = {"generated": 0, "copied": 0, "skipped": 0, "failed": 0}
    t0 = time.perf_counter()
    processed = 0

    for lang, clean_path in selected_cleans:
        clean_id = Path(clean_path).stem
        try:
            clean_audio, sr = _load_cached(clean_path)
        except Exception as e:
            print(f"[HATA] clean yüklenemedi {clean_path}: {e}")
            continue
        # Çok kısa clean dosyalarını mirror-pad ile uzat (STOI için)
        if len(clean_audio) / sr < DEFAULT_MIN_CLEAN_DURATION:
            clean_audio = extend_to_min_duration(
                clean_audio, sr, DEFAULT_MIN_CLEAN_DURATION, strategy="mirror"
            )
        clean_duration = len(clean_audio) / sr

        for scene, noise_path in noise_scenes.items():
            try:
                noise_audio, _ = _load_cached(noise_path)
            except Exception as e:
                print(f"[HATA] noise yüklenemedi {noise_path}: {e}")
                continue
            noise_duration = len(noise_audio) / sr
            offset = _deterministic_noise_offset(
                clean_id, scene, noise_duration, clean_duration, seed=args.seed
            )
            noise_slice = _slice_noise(noise_audio, sr, offset, len(clean_audio))

            for snr in snrs:
                if processed >= total_combos:
                    break
                processed += 1
                if progress is not None:
                    progress.update(1)

                fname = build_mix_filename(lang, clean_id, scene, float(snr))
                dest = os.path.join(out_dir, fname)

                # Idempotent: dosya + manifest satırı varsa atla
                if not args.force and fname in existing_filenames:
                    stats["skipped"] += 1
                    continue

                # Profil arası yeniden kullanım
                if not args.no_cross_reuse and not args.force:
                    src_path = find_existing_mix(fname, root=args.input_root)
                    if src_path and os.path.abspath(src_path) != os.path.abspath(dest):
                        try:
                            shutil.copy2(src_path, dest)
                            # achieved_snr ölçmek için yine de mix'i yükle
                            mixed_audio, _ = load_audio(dest)
                            achieved = _measure_achieved_snr(clean_audio, mixed_audio)
                            src_profile = Path(src_path).parent.name
                            entry = MixEntry(
                                idx=next_idx,
                                lang=lang,
                                clean_id=clean_id,
                                clean_path=clean_path.replace("\\", "/"),
                                noise_scene=scene,
                                noise_path=noise_path.replace("\\", "/"),
                                noise_offset_sec=round(offset, 3),
                                target_snr_db=float(snr),
                                achieved_snr_db=round(float(achieved), 3),
                                duration_sec=round(clean_duration, 3),
                                mix_filename=fname,
                                source=f"copied_from:{src_profile}",
                            )
                            entries.append(entry)
                            existing_filenames.add(fname)
                            next_idx += 1
                            stats["copied"] += 1
                            continue
                        except Exception as e:
                            print(f"[KOPYA HATA] {fname}: {e}; üretmeye geçiliyor")

                # Üret
                try:
                    mixed = mix_at_snr(clean_audio, noise_slice, float(snr))
                    save_audio(dest, mixed, sr=sr)
                    achieved = _measure_achieved_snr(clean_audio, mixed)
                    entry = MixEntry(
                        idx=next_idx,
                        lang=lang,
                        clean_id=clean_id,
                        clean_path=clean_path.replace("\\", "/"),
                        noise_scene=scene,
                        noise_path=noise_path.replace("\\", "/"),
                        noise_offset_sec=round(offset, 3),
                        target_snr_db=float(snr),
                        achieved_snr_db=round(float(achieved), 3),
                        duration_sec=round(clean_duration, 3),
                        mix_filename=fname,
                        source="generated",
                    )
                    entries.append(entry)
                    existing_filenames.add(fname)
                    next_idx += 1
                    stats["generated"] += 1
                except Exception as e:
                    print(f"[ÜRET HATA] {fname}: {e}")
                    stats["failed"] += 1

            if processed >= total_combos:
                break
        if processed >= total_combos:
            break

    if progress is not None:
        progress.close()

    # 5. Manifest'i yaz
    save_manifest(manifest_path, entries)

    elapsed = time.perf_counter() - t0
    print()
    print(f"=== {profile_name} özeti ({elapsed:.1f}s) ===")
    print(f"  Üretildi: {stats['generated']}")
    print(f"  Kopyalandı (profil arası): {stats['copied']}")
    print(f"  Atlandı (zaten var): {stats['skipped']}")
    print(f"  Başarısız: {stats['failed']}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Klasör: {out_dir}/")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
