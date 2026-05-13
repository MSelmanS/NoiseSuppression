"""Sentetik veri (clean + noise -> SNR seviyeleri) üzerinde performans + kalite benchmark'ı.

Dış döngü model, iç döngü (clean, noise) çifti x SNR -- böylece her model sadece
bir kez yüklenir. Sonuç: model x SNR pivotları, raw CSV ve metrik-vs-SNR grafikleri.

    python -m scripts.bench_synthetic \\
        --clean-dir data/clean --noise-dir data/noise \\
        --snrs -5 0 5 10 15 --max-pairs 20 --n-repeats 3 --models all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from glob import glob

# Direct çalıştırma (python scripts/bench_synthetic.py) için proje kökünü path'e ekle.
# `python -m scripts.bench_synthetic` ile çağrılırsa zaten doğru.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from audio_io.file_io import load_audio, save_audio
from benchmark.mixer import extend_to_min_duration
from benchmark.metrics import PeakRSSTracker, time_it, model_size_mb, param_count
from benchmark.runner import run_processing_only
from benchmark.report import (
    save_csv,
    save_xlsx,
    save_per_snr_summary,
    plot_per_snr,
    print_table,
)
from benchmark.sampling import pick_listening_samples, is_selected
from benchmark.anomaly import detect_anomalies, summarize_anomalies
from benchmark.html_report import generate_html_report
from benchmark.mix_manifest import (
    load_manifest,
    manifest_path_for,
    verify_manifest,
)
from scripts._model_registry import resolve_models, MODEL_REGISTRY
from scripts.profiles import PROFILES, get_profile, estimate_measurements


def _interactive_setup() -> dict:
    """Konsoldan interaktif konfigürasyon iste.

    F5 / Run button'dan başlatıldığında kullanıcı bir profil seçer ve
    save-strategy / HTML gibi yan seçenekleri belirler. Dönüş dict olarak
    argparse Namespace'e merge edilir.
    """
    print("=" * 50)
    print("  Denoiser Benchmark — İnteraktif Konfigürasyon")
    print("=" * 50)
    print()
    print("Mevcut profiller:")
    names = list(PROFILES.keys())
    for i, name in enumerate(names, start=1):
        desc = PROFILES[name].get("description", "")
        print(f"  [{i}] {name:<10}  {desc}")
    print(f"  [{len(names) + 1}] custom    Tüm parametreleri manuel gir")
    print()

    def _ask(prompt: str, default: str) -> str:
        try:
            ans = input(f"{prompt} [{default}]: ").strip()
        except EOFError:
            ans = ""
        return ans if ans else default

    # Profil seçimi
    choice = _ask(f"Profil seçimi [1-{len(names) + 1}]", "1")
    overrides: dict = {}
    try:
        idx = int(choice) - 1
    except ValueError:
        idx = 0
    if 0 <= idx < len(names):
        overrides["profile"] = names[idx]
    else:
        overrides["profile"] = None  # custom

    if overrides["profile"] is None:
        # Custom: tüm önemli parametreleri sor
        snrs = _ask("SNR seviyeleri (boşlukla)", "-5 0 5 10 15")
        overrides["snrs"] = [float(x) for x in snrs.split()]
        overrides["max_pairs"] = int(_ask("max-pairs", "20"))
        overrides["n_repeats"] = int(_ask("n-repeats", "3"))
        overrides["models"] = _ask("models", "all")
    else:
        # Profil seçildi; sadece nadiren değiştirilen seçenekleri sor
        pass

    overrides["save_strategy"] = _ask(
        "save-strategy [all/samples/none]", "samples"
    )
    html_choice = _ask("HTML rapor üretilsin mi? [Y/n]", "Y").lower()
    overrides["no_html_report"] = html_choice.startswith("n")

    print()
    print(f"Başlatılıyor: profile={overrides.get('profile') or 'custom'}, "
          f"save-strategy={overrides['save_strategy']}, "
          f"html={'evet' if not overrides['no_html_report'] else 'hayır'}")
    print()
    return overrides


def _require_manifest(profile: str | None, input_root: str = "input_data") -> str:
    """Manifest yolunu döndür; profile yoksa veya manifest yoksa exit."""
    if profile is None:
        print(
            "Hata: bench_synthetic --profile NAME ile çağrılmalı.\n"
            "Mevcut profiller için: scripts/profiles.py",
            file=sys.stderr,
        )
        sys.exit(2)
    mp = manifest_path_for(profile, root=input_root)
    if not os.path.isfile(mp):
        print(
            f"Hata: manifest yok: {mp}\n"
            f"Önce: python -m scripts.build_mixes --profile {profile}",
            file=sys.stderr,
        )
        sys.exit(2)
    entries, missing = verify_manifest(mp)
    if missing:
        print(
            f"Hata: manifest {len(entries)} dosya bekliyor ama "
            f"{len(missing)} eksik. Tekrar inşa et:",
            file=sys.stderr,
        )
        for m in missing[:5]:
            print(f"  - {m}", file=sys.stderr)
        print(
            f"\n  python -m scripts.build_mixes --profile {profile} --force",
            file=sys.stderr,
        )
        sys.exit(2)
    return mp


def _scene_from_path(noise_path: str) -> str:
    """data/noise/{SCENE}/ch01.wav -> {SCENE}"""
    parts = os.path.normpath(noise_path).split(os.sep)
    # parent of the .wav file
    return parts[-2] if len(parts) >= 2 else "unknown"


def _lang_from_path(clean_path: str) -> str:
    """data/clean/en/spkXX.wav -> en; data/clean/tr/... -> tr"""
    parts = os.path.normpath(clean_path).split(os.sep)
    for p in parts:
        if p.lower() in ("en", "tr"):
            return p.lower()
    return "xx"


# Profile yoksa kullanılacak yerleşik varsayılanlar
_BUILTIN_DEFAULTS: dict = {
    "clean_dir": "data/clean",
    "noise_dir": "data/noise",
    "snrs": [-5.0, 0.0, 5.0, 10.0, 15.0],
    "max_pairs": 20,
    "n_repeats": 3,
    "models": "all",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentetik karışım (clean+noise) üzerinde benchmark + kalite metrikleri.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Başlangıçta menüden profil ve seçenekleri sor (F5 / Run button için).",
    )
    # --profile profile dict'inden değerleri okur; aşağıdaki --xxx None default'ları
    # "kullanıcı bunu verdi mi?" ayrımı için kullanılır.
    p.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default=None,
        help=f"Önceden tanımlı profil. Geçerli: {','.join(PROFILES.keys())}",
    )
    p.add_argument(
        "--clean-dir",
        default=None,
        help="Temiz konuşma .wav klasörü (varsayılan: profil veya data/clean)",
    )
    p.add_argument(
        "--noise-dir",
        default=None,
        help="Gürültü .wav klasörü (varsayılan: profil veya data/noise)",
    )
    p.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=None,
        help="Hedef SNR seviyeleri (dB).",
    )
    p.add_argument(
        "--out-dir",
        default="output",
        help="Çıktı kök klasörü. Zaman damgalı alt klasör oluşturulur.",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="(clean, noise) çiftlerinin maks. sayısı (rastgele örnek).",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Her process() ölçümü için tekrar sayısı.",
    )
    p.add_argument(
        "--models",
        default=None,
        help=f"'all' veya virgülle ayrılmış isimler. Geçerli: {','.join(MODEL_REGISTRY.keys())}",
    )
    p.add_argument("--seed", type=int, default=42, help="Çift örnekleme tohumu.")
    p.add_argument(
        "--min-clean-duration",
        type=float,
        default=3.5,
        help=(
            "Clean wav minimum süresi (saniye). Bu eşikten kısa dosyalar için "
            "--short-clean-strategy davranışı uygulanır. STOI yaklaşık 3 sn'den "
            "kısa girdide 1e-5 fallback'i kullanır; STOI'nin anlamlı olmasını "
            "istiyorsak burayı yüksek tutmak gerek."
        ),
    )
    p.add_argument(
        "--short-clean-strategy",
        choices=["pad", "skip"],
        default="pad",
        help=(
            "Kısa clean dosyaları nasıl ele alalım:\n"
            "  pad  — aynalama ile min süreye uzat (data kaybı yok, varsayılan)\n"
            "  skip — pool'dan at"
        ),
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Her ölçüm önce yapılan ısınma çağrısını atla.",
    )
    p.add_argument(
        "--no-save-wavs",
        action="store_true",
        help="(Deprecated, --save-strategy=none ile aynı) Denoised + noisy mix kaydetme.",
    )
    p.add_argument(
        "--save-strategy",
        choices=["all", "samples", "none"],
        default="samples",
        help=(
            "Wav kayıt stratejisi:\n"
            "  all     — her (model,pair,snr) için kaydet (eski davranış, büyük disk)\n"
            "  samples — sadece dinleme galerisi için seçilen 28 + 5 örnek (varsayılan)\n"
            "  none    — hiçbir wav kaydetme, sadece metrikler\n"
        ),
    )
    p.add_argument(
        "--no-html-report",
        action="store_true",
        help="HTML raporu üretme (varsayılan: üret). save-strategy=none ise yine atlanır.",
    )
    # --use-prebuilt kaldırıldı: bench_synthetic her zaman pre-built mix kullanır.
    # Manifest yoksa `python -m scripts.build_mixes --profile X` ile üretilir.
    args = p.parse_args(argv)

    # İnteraktif mod: --interactive verildiyse VEYA profil belirtilmediyse,
    # konsoldan profil + seçenekleri sor.
    if args.interactive or args.profile is None:
        overrides = _interactive_setup()
        for key, value in overrides.items():
            setattr(args, key, value)

    # Profile + CLI override birleştir.
    # --profile verilmişse o profile başlat, yoksa _BUILTIN_DEFAULTS'a düş.
    if args.profile is not None:
        base = get_profile(args.profile)
        print(f"Profil seçildi: '{args.profile}' — {base.get('description', '')}")
    else:
        base = _BUILTIN_DEFAULTS

    # None olan her arg base'den; explicit verilen değerler dokunulmaz.
    overridable = ("clean_dir", "noise_dir", "snrs", "max_pairs", "n_repeats", "models")
    for key in overridable:
        if getattr(args, key) is None:
            setattr(args, key, base[key])

    # Tahmini koşum büyüklüğünü logla
    try:
        n_models = len(resolve_models(args.models))
    except Exception:
        n_models = len(MODEL_REGISTRY)
    n_total = estimate_measurements(
        {"max_pairs": args.max_pairs, "snrs": args.snrs, "n_repeats": args.n_repeats},
        n_models=n_models,
    )
    print(
        f"Konfig: {args.max_pairs} pair × {len(args.snrs)} SNR × "
        f"{args.n_repeats} rep × {n_models} model = {n_total} process çağrısı"
    )

    return args


def _list_wavs(d: str) -> list[str]:
    """Klasördeki .wav dosyalarını (alt klasörler dahil) sırala."""
    if not os.path.isdir(d):
        return []
    paths = sorted(glob(os.path.join(d, "**", "*.wav"), recursive=True))
    return paths


def _filter_by_duration(paths: list[str], min_seconds: float) -> tuple[list[str], list[str]]:
    """Wav listesini süresine göre ayır.

    pystoi STOI hesaplaması için ~3 s'lik konuşma şart; aksi halde
    'Not enough STFT frames' uyarısı verir ve 1e-5 döndürür. Kısa dosyaları
    erkenden eleriz, ölçümler temiz olur.

    Returns: (kept, skipped) — yan yana kalan ve atılan dosyalar.
    """
    import soundfile as sf
    kept: list[str] = []
    skipped: list[str] = []
    for p in paths:
        try:
            if sf.info(p).duration >= min_seconds:
                kept.append(p)
            else:
                skipped.append(p)
        except Exception:
            # Format hatası vs.; sessizce at
            skipped.append(p)
    return kept, skipped


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    clean_paths = _list_wavs(args.clean_dir)
    noise_paths = _list_wavs(args.noise_dir)
    if not clean_paths:
        print(f"Hata: {args.clean_dir} içinde .wav yok.", file=sys.stderr)
        return 2
    if not noise_paths:
        print(f"Hata: {args.noise_dir} içinde .wav yok.", file=sys.stderr)
        return 2

    # Çok kısa clean wav'lar STOI'yi düşürür (1e-5 fallback).
    # pad: aynalama ile uzat, skip: pool'dan at.
    if args.min_clean_duration > 0:
        if args.short_clean_strategy == "skip":
            clean_paths, skipped = _filter_by_duration(clean_paths, args.min_clean_duration)
            if skipped:
                print(
                    f"min-clean-duration={args.min_clean_duration:.1f}s eşiğinden "
                    f"{len(skipped)} clean dosyası atlandı:"
                )
                for p in skipped:
                    print(f"  - {p}")
            if not clean_paths:
                print(
                    f"Hata: --min-clean-duration={args.min_clean_duration:.1f}s sonrası "
                    "hiç clean dosyası kalmadı.",
                    file=sys.stderr,
                )
                return 2
        else:  # pad
            # Sadece hangi dosyalar pad edilecek bildiriyoruz; gerçek uzatma
            # _load_cached çağrısında yapılır (download yerine in-memory).
            from soundfile import info as sf_info
            short_files = []
            for p in clean_paths:
                try:
                    if sf_info(p).duration < args.min_clean_duration:
                        short_files.append(p)
                except Exception:
                    pass
            if short_files:
                print(
                    f"min-clean-duration={args.min_clean_duration:.1f}s altındaki "
                    f"{len(short_files)} dosya aynalama ile uzatılacak:"
                )
                for p in short_files:
                    print(f"  - {p}")

    try:
        model_classes = resolve_models(args.models)
    except ValueError as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 2

    # Pre-built mix yükle (memory mode tamamen kaldırıldı).
    manifest_path = _require_manifest(args.profile)
    entries = load_manifest(manifest_path)
    pair_map: dict[tuple[str, str], int] = {}
    unique_pairs: list[tuple[str, str]] = []
    snr_set: set[float] = set()
    prebuilt_lookup: dict[tuple[int, float], str] = {}
    for e in entries:
        key = (e.clean_path, e.noise_path)
        if key not in pair_map:
            pair_map[key] = len(unique_pairs)
            unique_pairs.append(key)
        pi = pair_map[key]
        snr_set.add(float(e.target_snr_db))
        prebuilt_lookup[(pi, float(e.target_snr_db))] = os.path.join(
            os.path.dirname(manifest_path), e.mix_filename
        )
    pairs = unique_pairs
    args.snrs = sorted(snr_set)
    args.max_pairs = len(pairs)
    print(
        f"Pre-built manifest: {len(entries)} mix "
        f"({len(pairs)} pair x {len(args.snrs)} SNR) — {manifest_path}"
    )

    print(f"Clean dosya: {len(clean_paths)}, Noise dosya: {len(noise_paths)}")
    print(f"Seçilen çift: {len(pairs)} (max-pairs={args.max_pairs}, seed={args.seed})")
    print(f"SNR seviyeleri: {args.snrs}")
    print(f"Modeller: {[m.__name__ for m in model_classes]}")
    print(f"Tekrar: {args.n_repeats}\n")

    # Tüm clean+noise ses dosyalarını önceden yükle (RAM tutarsa); aynı dosya
    # tekrar tekrar diskten okunmasın.
    cache: dict[str, "tuple"] = {}
    clean_paths_set = set(clean_paths)
    pad_active = (
        args.min_clean_duration > 0 and args.short_clean_strategy == "pad"
    )

    def _load_cached(path: str):
        if path not in cache:
            audio, sr = load_audio(path)
            # Clean ve kısaysa uzat. Noise için pad uygulanmaz; mix_at_snr'da
            # noise zaten clean uzunluğuna pad/trim ediliyor.
            if pad_active and path in clean_paths_set:
                duration = len(audio) / sr
                if duration < args.min_clean_duration:
                    audio = extend_to_min_duration(
                        audio, sr, args.min_clean_duration, strategy="mirror"
                    )
            cache[path] = (audio, sr)
        return cache[path]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"bench_synthetic_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Çıktı klasörü: {out_dir}\n")

    rows: list[dict] = []
    do_warmup = not args.no_warmup

    # Geriye uyumluluk: --no-save-wavs varsa --save-strategy none kabul et
    if args.no_save_wavs:
        args.save_strategy = "none"
    print(f"Save strategy: {args.save_strategy}")

    # Strategy='samples' ise hangi (pi, snr) kombinasyonlarının kaydedileceğini
    # şimdi belirle (deterministik, seed=42).
    selected_samples: set[tuple[int, float]] = set()
    if args.save_strategy == "samples":
        selected_samples = pick_listening_samples(pairs, args.snrs, seed=args.seed)
        print(
            f"Dinleme galerisi seçimi: {len(selected_samples)} (pair, snr) kombinasyonu "
            f"(her biri için {len(model_classes)} model çıktısı kaydedilecek)"
        )

    # Klasör hazırlığı strategy'ye göre
    # - all: out_dir/_input/ + out_dir/{model}/
    # - samples: out_dir/samples/_reference/ + out_dir/samples/{model}/
    # - none: hiçbiri
    if args.save_strategy == "all":
        ref_dir = os.path.join(out_dir, "_input")
        per_model_root = out_dir
    elif args.save_strategy == "samples":
        ref_dir = os.path.join(out_dir, "samples", "_reference")
        per_model_root = os.path.join(out_dir, "samples")
    else:  # none
        ref_dir = ""
        per_model_root = ""

    if args.save_strategy != "none":
        os.makedirs(ref_dir, exist_ok=True)

    saved_inputs: set[tuple[int, float]] = set()

    # Dış döngü: model (load bir kez)
    for mi, model_class in enumerate(model_classes, start=1):
        print(f"[{mi}/{len(model_classes)}] Model yükleniyor: {model_class.__name__}")
        try:
            with PeakRSSTracker() as load_tracker:
                model = model_class()
                _, load_time = time_it(model.load)
            inner = model.model if hasattr(model, "model") else None
            size_mb = round(model_size_mb(inner), 2)
            n_params = param_count(inner)
            load_peak_mb = round(load_tracker.peak_mb, 1)
            print(
                f"    yükleme: {load_time:.2f}s, "
                f"peak_RAM(load)={load_peak_mb}MB, boyut={size_mb}MB, params={n_params}"
            )
        except Exception as e:
            print(f"    YÜKLEME HATASI: {e}")
            continue

        # İç döngü: (clean, noise) çifti x snr
        total = len(pairs) * len(args.snrs)
        done = 0
        t0 = time.perf_counter()
        for pi, (clean_path, noise_path) in enumerate(pairs):
            try:
                clean, clean_sr = _load_cached(clean_path)
                noise, _ = _load_cached(noise_path)
            except Exception as e:
                print(f"    Pair atlandı ({clean_path} + {noise_path}): {e}")
                continue

            for snr in args.snrs:
                done += 1
                try:
                    # Pre-built mix dosyasını diskten oku (manifest garantili).
                    mix_path = prebuilt_lookup.get((pi, float(snr)))
                    if not mix_path:
                        raise RuntimeError(
                            f"manifest'te (pair={pi}, snr={snr}) için mix yok"
                        )
                    noisy, _ = _load_cached(mix_path)

                    snr_tag = (
                        f"{int(snr):+d}dB" if float(snr).is_integer() else f"{snr:+.1f}dB"
                    )
                    scene = _scene_from_path(noise_path)
                    lang = _lang_from_path(clean_path)

                    # Bu (pair, snr) için clean + noisy mix'i ilk kez görüyorsak kaydet.
                    # samples strategy'sinde sadece seçilen pair/snr'lar için.
                    should_save_input = (
                        args.save_strategy == "all"
                        or (args.save_strategy == "samples"
                            and is_selected(pi, snr, selected_samples))
                    )
                    if should_save_input and (pi, snr) not in saved_inputs:
                        if args.save_strategy == "all":
                            clean_name = f"pair{pi:02d}_clean.wav"
                            noisy_name = f"pair{pi:02d}_snr{snr_tag}_noisy.wav"
                        else:  # samples
                            clean_name = f"clean_pair{pi:02d}_{lang}.wav"
                            noisy_name = f"noisy_{scene}_snr{snr_tag}.wav"
                        save_audio(
                            os.path.join(ref_dir, clean_name),
                            clean,
                            sr=model.sample_rate,
                        )
                        save_audio(
                            os.path.join(ref_dir, noisy_name),
                            noisy,
                            sr=model.sample_rate,
                        )
                        saved_inputs.add((pi, snr))

                    # Process ölçümü -- RAM'i de pencere ile izle (per-pair)
                    with PeakRSSTracker() as proc_tracker:
                        proc_result, denoised = run_processing_only(
                            model,
                            noisy,
                            reference=clean,
                            n_repeats=args.n_repeats,
                            do_warmup=do_warmup,
                        )

                    # Modelin çıktısını kaydet (strategy'ye göre).
                    should_save_output = (
                        denoised is not None
                        and (
                            args.save_strategy == "all"
                            or (args.save_strategy == "samples"
                                and is_selected(pi, snr, selected_samples))
                        )
                    )
                    if should_save_output:
                        model_out_dir = os.path.join(per_model_root, model.name)
                        os.makedirs(model_out_dir, exist_ok=True)
                        if args.save_strategy == "all":
                            out_name = f"pair{pi:02d}_snr{snr_tag}.wav"
                        else:  # samples
                            out_name = f"{scene}_snr{snr_tag}.wav"
                        save_audio(
                            os.path.join(model_out_dir, out_name),
                            denoised,
                            sr=model.sample_rate,
                        )
                    row = {
                        "model": model.name,
                        "snr_db": snr,
                        # Sahne adı analiz için lazım; clean dilini de saklayalım
                        "scene": scene,
                        "lang": lang,
                        "clean_file": os.path.basename(clean_path),
                        "noise_file": f"{scene}/{os.path.basename(noise_path)}",
                        "load_time_s": round(load_time, 3),
                        **proc_result,
                        # load + process peak'inin maks'ı (yaklaşık peak_ram benzeri)
                        "peak_ram_mb": round(max(load_peak_mb, proc_tracker.peak_mb), 1),
                        "model_size_mb": size_mb,
                        "param_count": n_params,
                    }
                    rows.append(row)
                except Exception as e:
                    print(
                        f"    pair#{pi+1} snr={snr} HATA: {e}"
                    )

            if (pi + 1) % 5 == 0 or (pi + 1) == len(pairs):
                elapsed = time.perf_counter() - t0
                print(
                    f"    pair {pi+1}/{len(pairs)} ({done}/{total} ölçüm, {elapsed:.1f}s)"
                )

        # Model serbest -- bir sonraki model için RAM'i temizle
        del model
        import gc
        gc.collect()

    if not rows:
        print("Hiç sonuç üretilmedi.", file=sys.stderr)
        return 1

    # Raw + özet + grafikler
    csv_path = os.path.join(out_dir, "results_raw.csv")
    raw_xlsx = os.path.join(out_dir, "results_raw.xlsx")
    summary_xlsx = os.path.join(out_dir, "results_per_snr.xlsx")
    save_csv(rows, csv_path)
    save_xlsx(rows, raw_xlsx)
    save_per_snr_summary(rows, summary_xlsx)

    # Grafikler: hangi metrikler varsa onları çiz
    plots = [
        ("rtf_mean", "RTF", "plot_rtf_vs_snr.png"),
        ("si_sdr", "SI-SDR (dB)", "plot_sisdr_vs_snr.png"),
        ("stoi", "STOI", "plot_stoi_vs_snr.png"),
        ("pesq", "PESQ", "plot_pesq_vs_snr.png"),
        ("peak_ram_mb", "Peak RAM (MB)", "plot_peakram_vs_snr.png"),
    ]
    for metric, label, fname in plots:
        if any(metric in r for r in rows):
            plot_per_snr(rows, metric, os.path.join(out_dir, fname), ylabel=label)

    # Konsola kısa özet: model x mean kalite
    print("\n=== Model başına ortalama (tüm SNR/çiftler) ===")
    import pandas as pd

    df = pd.DataFrame(rows)
    agg_cols = [c for c in ("rtf_mean", "peak_ram_mb", "si_sdr", "stoi", "pesq") if c in df.columns]
    summary = df.groupby("model")[agg_cols].mean().round(3).reset_index()
    print_table(summary.to_dict(orient="records"))

    # Anomali yakalama (Task 4)
    anomalies = detect_anomalies(df)
    anomalies_path = os.path.join(out_dir, "anomalies.csv")
    if len(anomalies) > 0:
        anomalies.to_csv(anomalies_path, index=False, encoding="utf-8")
        counts = summarize_anomalies(anomalies)
        print(f"\n=== {len(anomalies)} anomali yakalandı ===")
        for atype, n in counts.items():
            print(f"  {atype}: {n}")
        print(f"  Detay: {anomalies_path}")
    else:
        print("\n=== Anomali yakalanmadı ===")

    # HTML rapor (Task 5)
    # save-strategy=none ise samples klasörü yok, galeri boş kalır; yine de
    # liderlik + hipotez + anomali + plot bölümleri anlamlı.
    if not args.no_html_report:
        try:
            report_path = generate_html_report(
                out_dir,
                rows,
                profile=args.profile,
            )
            size_mb = os.path.getsize(report_path) / (1024 * 1024)
            print(f"HTML rapor: {report_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"HTML rapor üretilemedi: {type(e).__name__}: {e}")

    print(f"\nRaw CSV     : {csv_path}")
    print(f"Raw XLSX    : {raw_xlsx}")
    print(f"Per-SNR XLSX: {summary_xlsx}")
    print(f"Grafikler   : {out_dir}/plot_*.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
