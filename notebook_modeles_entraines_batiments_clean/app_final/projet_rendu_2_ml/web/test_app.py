"""Tests simples et reproductibles de l'interface Flask locale."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from app import PROJECT_DIR, create_app
from image_processing import FEATURE_COUNT, image_to_features


DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "Dataset final-20260702T055807Z-3-001"
    / "Dataset final sans doublon"
)


def first_image(class_name: str) -> Path:
    return sorted((DATASET_DIR / class_name).glob("*.jpg"), key=lambda path: path.name)[0]


def upload(path: Path, filename: str | None = None):
    return io.BytesIO(path.read_bytes()), filename or path.name


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_preprocessing_has_1024_normalized_values(self):
        features = image_to_features(first_image("Gothique"))
        self.assertEqual(FEATURE_COUNT, len(features))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features))

    def test_manifest_lists_the_four_new_models(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        for name in ("Perceptron", "MLP", "RBF", "SVM"):
            self.assertIn(f"{name} — bâtiments 3 classes".encode(), response.data)
        self.assertIn(b'value="mlp_v2"', response.data)
        self.assertNotIn(b'value="mlp_v1"', response.data)

    def test_real_single_upload(self):
        response = self.client.post(
            "/",
            data={"model": "perceptron_v1", "images": upload(first_image("Art déco"))},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code)
        self.assertIn(b"margin", response.data)

    def test_multiple_upload_and_temporary_cleanup(self):
        before = set(Path(tempfile.gettempdir()).glob("ml_web_*"))
        response = self.client.post(
            "/",
            data={
                "model": "perceptron_v1",
                "images": [
                    upload(first_image("Art déco"), "art_deco.jpg"),
                    upload(first_image("Gothique"), "gothique.jpg"),
                ],
            },
            content_type="multipart/form-data",
        )
        after = set(Path(tempfile.gettempdir()).glob("ml_web_*"))
        self.assertEqual(200, response.status_code)
        self.assertIn(b"art_deco.jpg", response.data)
        self.assertIn(b"gothique.jpg", response.data)
        self.assertEqual(before, after)

    def test_invalid_image(self):
        response = self.client.post(
            "/",
            data={
                "model": "perceptron_v1",
                "images": (io.BytesIO(b"ceci n'est pas une image"), "invalide.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code)
        self.assertIn("image invalide".encode(), response.data)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory(prefix="ml_missing_models_") as empty_models:
            empty_models_path = Path(empty_models)
            manifest = {
                "feature_count": 1024,
                "image_size": [32, 32],
                "classes": {"0": "Art déco", "1": "Art nouveau", "2": "Gothique"},
                "models": [{
                    "id": "perceptron_v1",
                    "display_name": "Perceptron — bâtiments 3 classes",
                    "algorithm": "Perceptron",
                    "file": "modele_absent.model",
                    "feature_count": 1024,
                    "class_count": 3,
                    "score_type": "margin",
                    "test_accuracy": 0.0,
                }],
            }
            (empty_models_path / "models_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            app = create_app({"TESTING": True, "MODELS_DIR": empty_models_path})
            response = app.test_client().post(
                "/",
                data={"model": "perceptron_v1", "images": upload(first_image("Art déco"))},
                content_type="multipart/form-data",
            )
            self.assertEqual(200, response.status_code)
            self.assertIn("est absent".encode(), response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
