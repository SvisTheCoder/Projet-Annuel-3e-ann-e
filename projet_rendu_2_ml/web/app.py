"""Interface web locale : upload d'images puis prédiction par le C++."""

import base64
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

from image_processing import ALLOWED_EXTENSIONS, FEATURE_COUNT, image_to_features


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
MANIFEST_FILENAME = "models_manifest.json"
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_COUNT = 10


def find_predict_cli() -> Path:
    configured_path = os.environ.get("PREDICT_CLI_PATH")
    candidates = [
        Path(configured_path) if configured_path else None,
        PROJECT_DIR / "cmake-build-debug" / "predict_cli.exe",
        PROJECT_DIR / "build" / "predict_cli.exe",
        PROJECT_DIR / "build" / "Release" / "predict_cli.exe",
        PROJECT_DIR / "build" / "predict_cli",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "predict_cli est introuvable. Compilez d'abord la cible CMake predict_cli."
    )


def load_model_catalog(models_dir: Path) -> tuple[list[dict], list[str]]:
    """Charge le manifest et garde uniquement les modèles compatibles présents."""
    manifest_path = models_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return [], [f"Manifest des modèles absent : {manifest_path}."]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [], [f"Manifest des modèles invalide : {error}."]

    classes = manifest.get("classes", {})
    image_size = manifest.get("image_size", [])
    if manifest.get("feature_count") != FEATURE_COUNT or image_size != [32, 32]:
        return [], ["Le manifest n'est pas compatible avec les images 32 × 32."]
    if set(classes) != {"0", "1", "2"}:
        return [], ["Le manifest doit déclarer exactement les classes 0, 1 et 2."]

    available_models = []
    errors = []
    for entry in manifest.get("models", []):
        try:
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("nom de fichier non sécurisé")
            model_path = models_dir / filename
            if not model_path.is_file():
                errors.append(f"Le fichier du modèle {entry.get('display_name', filename)} est absent.")
                continue
            if entry.get("feature_count") != FEATURE_COUNT or entry.get("class_count") != 3:
                errors.append(f"Le modèle {entry.get('display_name', filename)} est incompatible.")
                continue

            available_models.append(
                {
                    **entry,
                    "path": model_path,
                    "image_size_text": "32 × 32",
                    "class_names": [classes[str(index)] for index in range(3)],
                    "score_label": (
                        "marge"
                        if entry.get("score_type") == "margin"
                        else "meilleure probabilité one-vs-rest"
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"Entrée de modèle invalide dans le manifest : {error}.")

    if not available_models:
        errors.append("Aucun modèle compatible n'est disponible.")
    return available_models, errors


def create_preview(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        corrected = ImageOps.exif_transpose(image)
        preview = corrected.convert("RGB")
        preview.thumbnail((320, 220), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        preview.save(output, format="JPEG", quality=82)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def write_features(path: Path, features: list[float]) -> None:
    if len(features) != FEATURE_COUNT:
        raise ValueError("Le prétraitement n'a pas produit 1024 valeurs.")
    path.write_text("\n".join(f"{value:.17g}" for value in features), encoding="utf-8")


def run_prediction(
    predict_cli: Path,
    model_path: Path,
    features_path: Path,
    expected_class_count: int,
) -> dict:
    completed = subprocess.run(
        [
            str(predict_cli),
            str(model_path),
            str(features_path),
            "--expected-class-count",
            str(expected_class_count),
        ],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or (
            f"predict_cli s'est arrêté avec le code {completed.returncode}."
        )
        raise RuntimeError(message)
    if completed.stderr.strip():
        raise RuntimeError(completed.stderr.strip())
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("predict_cli n'a pas retourné un JSON valide.") from error


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_IMAGE_COUNT * MAX_IMAGE_SIZE,
        MODELS_DIR=DEFAULT_MODELS_DIR,
    )
    if test_config:
        app.config.update(test_config)

    def page_context(selected_model: str = "", results=None, errors=None):
        models, catalog_errors = load_model_catalog(Path(app.config["MODELS_DIR"]))
        if not selected_model and models:
            selected_model = models[0]["id"]
        return {
            "models": models,
            "selected_model": selected_model,
            "results": results or [],
            "errors": [*catalog_errors, *(errors or [])],
        }

    @app.errorhandler(413)
    def request_too_large(_error):
        return render_template(
            "index.html",
            **page_context(errors=["La requête dépasse la taille maximale autorisée."]),
        ), 413

    @app.route("/", methods=["GET", "POST"])
    def index():
        models_dir = Path(app.config["MODELS_DIR"])
        models, catalog_errors = load_model_catalog(models_dir)
        selected_model = request.form.get(
            "model",
            models[0]["id"] if models else "",
        )
        results = []
        errors = list(catalog_errors)

        if request.method == "POST":
            model_entry = next(
                (model for model in models if model["id"] == selected_model),
                None,
            )
            if model_entry is None:
                errors.append("Le modèle sélectionné est absent, invalide ou incompatible.")

            uploads = [
                upload
                for upload in request.files.getlist("images")
                if upload.filename
            ]
            if not uploads:
                errors.append("Sélectionnez au moins une image.")
            elif len(uploads) > MAX_IMAGE_COUNT:
                errors.append(f"Maximum {MAX_IMAGE_COUNT} images par requête.")

            if model_entry is not None and uploads and len(uploads) <= MAX_IMAGE_COUNT:
                try:
                    predict_cli = find_predict_cli()
                except FileNotFoundError as error:
                    errors.append(str(error))
                    predict_cli = None

                if predict_cli is not None:
                    with tempfile.TemporaryDirectory(prefix="ml_web_") as temp_directory:
                        temp_path = Path(temp_directory)
                        for index, upload in enumerate(uploads):
                            safe_name = secure_filename(upload.filename)
                            suffix = Path(safe_name).suffix.lower()
                            if suffix not in ALLOWED_EXTENSIONS:
                                errors.append(
                                    f"{safe_name or 'Fichier'} : format non accepté "
                                    "(PNG, JPG ou JPEG uniquement)."
                                )
                                continue

                            image_bytes = upload.stream.read(MAX_IMAGE_SIZE + 1)
                            if len(image_bytes) > MAX_IMAGE_SIZE:
                                errors.append(f"{safe_name} : fichier supérieur à 10 Mo.")
                                continue

                            try:
                                features = image_to_features(io.BytesIO(image_bytes))
                                preview = create_preview(image_bytes)
                                features_path = temp_path / f"features_{index}.txt"
                                write_features(features_path, features)
                                prediction = run_prediction(
                                    predict_cli,
                                    model_entry["path"],
                                    features_path,
                                    model_entry["class_count"],
                                )
                                results.append(
                                    {
                                        "filename": safe_name,
                                        "preview": preview,
                                        "display_name": model_entry["display_name"],
                                        **prediction,
                                    }
                                )
                            except (UnidentifiedImageError, OSError, ValueError) as error:
                                errors.append(f"{safe_name} : image invalide ({error}).")
                            except (RuntimeError, subprocess.TimeoutExpired) as error:
                                errors.append(f"{safe_name} : prédiction impossible ({error}).")

        return render_template(
            "index.html",
            models=models,
            selected_model=selected_model,
            results=results,
            errors=errors,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
