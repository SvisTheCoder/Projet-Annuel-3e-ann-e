"""Orchestre train_cli et construit le manifest, sans implémenter de ML."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


MODELS = (
    {
        "id": "perceptron_v1",
        "cli_name": "perceptron",
        "display_name": "Perceptron — bâtiments 3 classes",
        "algorithm": "Perceptron",
        "file": "buildings_3classes_32x32_perceptron_v1.model",
        "score_type": "margin",
    },
    {
        "id": "mlp_v2",
        "cli_name": "mlp",
        "display_name": "MLP — bâtiments 3 classes",
        "algorithm": "MLP",
        "file": "buildings_3classes_32x32_mlp_v2.model",
        "score_type": "best_ovr_probability",
        "cli_arguments": [
            "--epochs", "12",
            "--learning-rate", "0.02",
            "--hidden-size", "32",
        ],
    },
    {
        "id": "rbf_v1",
        "cli_name": "rbf",
        "display_name": "RBF — bâtiments 3 classes",
        "algorithm": "RBF",
        "file": "buildings_3classes_32x32_rbf_v1.model",
        "score_type": "best_ovr_probability",
    },
    {
        "id": "svm_v1",
        "cli_name": "svm",
        "display_name": "SVM — bâtiments 3 classes",
        "algorithm": "SVM",
        "file": "buildings_3classes_32x32_svm_v1.model",
        "score_type": "margin",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Entraîne les quatre modèles C++.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=project / "data" / "batiments_3_classes.csv",
    )
    parser.add_argument(
        "--train-cli",
        type=Path,
        default=project / "cmake-build-debug" / "train_cli.exe",
    )
    parser.add_argument("--models-dir", type=Path, default=project / "models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_model(
    train_cli: Path,
    csv_path: Path,
    models_dir: Path,
    seed: int,
    definition: dict,
) -> dict:
    output_path = models_dir / definition["file"]
    reports_dir = models_dir / "reports"
    report_path = reports_dir / f"{definition['id']}.json"
    reports_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(train_cli),
        "--model", definition["cli_name"],
        "--csv", str(csv_path),
        "--output", str(output_path),
        "--seed", str(seed),
        "--report", str(report_path),
    ]
    command.extend(definition.get("cli_arguments", []))
    print("\n> " + " ".join(f'"{item}"' if " " in item else item for item in command))
    completed = subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"Échec de train_cli pour {definition['display_name']} "
            f"(code {completed.returncode})."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("reload_validation") or not output_path.is_file():
        raise RuntimeError(f"Validation incomplète pour {definition['display_name']}.")

    return {
        "id": definition["id"],
        "display_name": definition["display_name"],
        "algorithm": definition["algorithm"],
        "file": definition["file"],
        "feature_count": report["feature_count"],
        "class_count": report["class_count"],
        "score_type": definition["score_type"],
        "train_size": report["train_size"],
        "test_size": report["test_size"],
        "train_accuracy": report["train_accuracy"],
        "test_accuracy": report["test_accuracy"],
        "confusion_matrix": report["confusion_matrix"],
        "parameters": report["parameters"],
        "duration_seconds": report["duration_seconds"],
        "model_size_bytes": report["model_size_bytes"],
        "reload_validation": report["reload_validation"],
        "created_at": report["created_at"],
    }


def main() -> int:
    arguments = parse_arguments()
    csv_path = arguments.csv.resolve()
    train_cli = arguments.train_cli.resolve()
    models_dir = arguments.models_dir.resolve()

    if not csv_path.is_file():
        print(f"CSV absent : {csv_path}", file=sys.stderr)
        return 2
    if not train_cli.is_file():
        print(f"train_cli absent : {train_cli}", file=sys.stderr)
        return 2

    models_dir.mkdir(parents=True, exist_ok=True)
    csv_sha256 = sha256_file(csv_path)
    model_entries = []

    try:
        for definition in MODELS:
            entry = train_model(
                train_cli,
                csv_path,
                models_dir,
                arguments.seed,
                definition,
            )
            model_entries.append(entry)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    manifest = {
        "dataset_name": "Buildings local dataset without duplicates",
        "dataset_file": csv_path.name,
        "dataset_sha256": csv_sha256,
        "feature_count": 1024,
        "image_size": [32, 32],
        "classes": {
            "0": "Art déco",
            "1": "Art nouveau",
            "2": "Gothique",
        },
        "models": model_entries,
    }

    manifest_path = models_dir / "models_manifest.json"
    temporary_manifest = models_dir / "models_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"\nManifest créé : {manifest_path}")
    print(f"CSV SHA-256 : {csv_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
