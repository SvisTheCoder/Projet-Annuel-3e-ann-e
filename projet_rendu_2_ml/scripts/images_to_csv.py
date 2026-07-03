"""Génère le CSV canonique à partir des trois dossiers d'images locaux."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"
sys.path.insert(0, str(WEB_DIR))

from image_processing import ALLOWED_EXTENSIONS, FEATURE_COUNT, image_to_features


CLASSES = (
    ("Art déco", 0),
    ("Art nouveau", 1),
    ("Gothique", 2),
)


def find_images(class_directory: Path) -> list[Path]:
    """Retourne les images dans un ordre stable et reproductible."""
    return sorted(
        (
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def generate_csv(dataset_directory: Path, output_csv: Path) -> Counter:
    distribution: Counter = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        for class_name, label in CLASSES:
            class_directory = dataset_directory / class_name
            if not class_directory.is_dir():
                raise FileNotFoundError(f"Dossier de classe absent : {class_directory}")

            images = find_images(class_directory)
            if not images:
                raise ValueError(f"Aucune image trouvée pour la classe {class_name}.")

            for image_path in images:
                features = image_to_features(image_path)
                writer.writerow([*features, label])
                distribution[label] += 1

    return distribution


def print_csv_summary(output_csv: Path, distribution: Counter) -> None:
    line_count = sum(distribution.values())
    labels = sorted(distribution)

    print(f"CSV : {output_csv}")
    print(f"Nombre de lignes : {line_count}")
    print(f"Nombre de colonnes : {FEATURE_COUNT + 1}")
    print(f"Nombre de features : {FEATURE_COUNT}")
    print(f"Labels présents : {', '.join(str(label) for label in labels)}")
    for class_name, label in CLASSES:
        print(f"{class_name} ({label}) : {distribution[label]}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convertit les dossiers Art déco, Art nouveau et Gothique en CSV."
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        distribution = generate_csv(
            arguments.dataset_directory.resolve(),
            arguments.output_csv.resolve(),
        )
        print_csv_summary(arguments.output_csv.resolve(), distribution)
        return 0
    except (OSError, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
