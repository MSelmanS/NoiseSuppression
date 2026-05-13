"""Gerçek (kayıtlı) wav üzerinde performans benchmark'ı.

Kalite metriği hesaplanmaz (temiz referans yok). GUI yok — argparse ile çağrılır:

    python -m scripts.bench_real --wav path/to/in.wav
    python -m scripts.bench_real --wav in.wav --out-dir output --n-repeats 3 --models all
    python -m scripts.bench_real --wav in.wav --models rnnoise,deepfilternet
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from glob import glob

# Direct çalıştırma için proje kökünü path'e ekle (python scripts/bench_real.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from audio_io.file_io import load_audio, save_audio
from benchmark.runner import run_model
from benchmark.report import save_csv, save_xlsx, print_table
from scripts._model_registry import resolve_models, MODEL_REGISTRY


# Run düğmesine basıldığında --wav verilmezse aranacak yerler (sıralı):
# ilk bulunan .wav kullanılır. Yeni bir varsayılan eklemek için listeye yaz.
_DEFAULT_WAV_SEARCH = [
    "test/*.wav",
    "data/clean/*.wav",
    "data/sample.wav",
]


def _find_default_wav() -> str | None:
    for pattern in _DEFAULT_WAV_SEARCH:
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tek bir wav dosyası üzerinde performans benchmark'ı.",
    )
    p.add_argument(
        "--wav",
        default=None,
        help=(
            ".wav giriş dosyası. Verilmezse sırayla test/, data/clean/ klasörlerinden "
            "ilk bulduğunu kullanır (Run düğmesinden çalıştırma kolaylığı)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default="output",
        help="Çıktı kök klasörü. Her çalıştırma zaman damgalı alt klasöre yazar.",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Her model için process() tekrar sayısı (mean/std hesaplamak için).",
    )
    p.add_argument(
        "--models",
        default="all",
        help=f"'all' veya virgülle ayrılmış isim listesi. Geçerli: {','.join(MODEL_REGISTRY.keys())}",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Warmup çağrısını atla (ilk çağrı dahil ölçülsün).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.wav is None:
        args.wav = _find_default_wav()
        if args.wav is None:
            print(
                "Hata: --wav verilmedi ve varsayılan konumlarda .wav bulunamadı "
                f"({_DEFAULT_WAV_SEARCH}).",
                file=sys.stderr,
            )
            return 2
        print(f"(--wav verilmedi, otomatik seçildi: {args.wav})")

    if not os.path.isfile(args.wav):
        print(f"Hata: {args.wav} bulunamadı.", file=sys.stderr)
        return 2

    try:
        model_classes = resolve_models(args.models)
    except ValueError as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 2

    print(f"Giriş: {args.wav}")
    audio, sr = load_audio(args.wav)
    print(f"Ses uzunluğu: {len(audio) / sr:.2f} sn ({len(audio)} sample, {sr} Hz)\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"bench_real_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Çıktı klasörü: {out_dir}\n")

    # Orijinali de yedekle (kulak testi referansı)
    save_audio(os.path.join(out_dir, "00_original.wav"), audio)

    results: list[dict] = []
    do_warmup = not args.no_warmup

    for i, model_class in enumerate(model_classes, start=1):
        print(f"[{i}/{len(model_classes)}] {model_class.__name__} çalıştırılıyor...")
        try:
            result, denoised = run_model(
                model_class,
                audio,
                reference=None,
                n_repeats=args.n_repeats,
                do_warmup=do_warmup,
            )
            results.append(result)

            out_wav = os.path.join(out_dir, f"{i:02d}_{result['model']}.wav")
            save_audio(out_wav, denoised)
            print(
                f"    OK  -> RTF={result['rtf_mean']}±{result['rtf_std']}, "
                f"süre={result['process_time_mean_s']}s, peak_RAM={result['peak_ram_mb']}MB"
            )
        except Exception as e:
            print(f"    HATA: {e}")

    if results:
        print_table(results)
        csv_path = os.path.join(out_dir, "results.csv")
        xlsx_path = os.path.join(out_dir, "results.xlsx")
        save_csv(results, csv_path)
        save_xlsx(results, xlsx_path)
        print(f"\nSonuçlar CSV : {csv_path}")
        print(f"Sonuçlar XLSX: {xlsx_path}")
        print(f"Temizlenmiş sesler: {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
