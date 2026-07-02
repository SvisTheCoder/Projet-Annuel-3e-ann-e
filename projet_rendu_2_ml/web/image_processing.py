"""Prétraitement unique utilisé par le CSV et l'interface web."""

from PIL import Image, ImageOps


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
FEATURE_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_to_features(image_source) -> list[float]:
    """Convertit une image en 1024 pixels gris normalisés, ligne par ligne."""
    with Image.open(image_source) as image:
        corrected = ImageOps.exif_transpose(image)
        grayscale = corrected.convert("L")
        resized = grayscale.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        if hasattr(resized, "get_flattened_data"):
            pixels = resized.get_flattened_data()
        else:
            pixels = resized.getdata()
        features = [float(pixel) / 255.0 for pixel in pixels]

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"Prétraitement invalide : {len(features)} valeurs au lieu de {FEATURE_COUNT}."
        )
    if any(value < 0.0 or value > 1.0 for value in features):
        raise ValueError("Prétraitement invalide : un pixel est hors de [0, 1].")

    return features
