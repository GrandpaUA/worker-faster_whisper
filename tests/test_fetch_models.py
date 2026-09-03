import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_fetch_models():
    spec = importlib.util.spec_from_file_location(
        "fetch_models_under_test",
        ROOT / "builder" / "fetch_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FetchModelsTests(unittest.TestCase):
    def test_default_whisper_prefetch_matches_production_model(self):
        module = load_fetch_models()

        self.assertEqual(module.parse_whisper_models(None), ["large-v2"])

    def test_whisper_models_accepts_comma_separated_override(self):
        module = load_fetch_models()

        self.assertEqual(module.parse_whisper_models("small, large-v2"), ["small", "large-v2"])

    def test_empty_whisper_models_is_loud_error(self):
        module = load_fetch_models()

        with self.assertRaisesRegex(ValueError, "WHISPER_MODELS"):
            module.parse_whisper_models(" , ")


if __name__ == "__main__":
    unittest.main()
