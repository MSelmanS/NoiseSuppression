"""Hem bench_real hem bench_synthetic'in paylaştığı model kayıt defteri.

CLI'da --models all veya virgülle ayrılmış isim listesi alıyoruz, buradan
sınıfa map ediyoruz. Yeni bir model eklenince sadece burası güncellenir.
"""

from models.spectral_subtraction import SpectralSubtraction
from models.rnnoise_model import RNNoiseDenoiser
from models.deepfilternet_model import DeepFilterNetDenoiser
from models.demucs_model import DemucsDns48, DemucsDns64, DemucsMaster64
from models.metricgan_model import MetricGANDenoiser


# İsim -> sınıf (CLI'dan --models seçimi bu isimlere göre)
MODEL_REGISTRY: dict[str, type] = {
    "spectral_subtraction": SpectralSubtraction,
    "rnnoise": RNNoiseDenoiser,
    "deepfilternet": DeepFilterNetDenoiser,
    "demucs_dns48": DemucsDns48,
    "demucs_dns64": DemucsDns64,
    "demucs_master64": DemucsMaster64,
    "metricgan_plus": MetricGANDenoiser,
}


def resolve_models(spec: str) -> list[type]:
    """CLI string'inden sınıf listesine dönüş.

    spec="all" -> hepsi (kayıt sırasıyla).
    Aksi halde virgülle ayrılmış isimler (örn. "rnnoise,deepfilternet").
    Bilinmeyen isim ValueError fırlatır.
    """
    if spec.strip().lower() == "all":
        return list(MODEL_REGISTRY.values())

    names = [n.strip() for n in spec.split(",") if n.strip()]
    classes: list[type] = []
    unknown: list[str] = []
    for name in names:
        if name in MODEL_REGISTRY:
            classes.append(MODEL_REGISTRY[name])
        else:
            unknown.append(name)
    if unknown:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Bilinmeyen model(ler): {unknown}. Geçerli: {valid}")
    return classes
