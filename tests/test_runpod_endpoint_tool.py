import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "runpod_endpoint_under_test",
        ROOT / "tools" / "runpod_endpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.modules.pop(spec.name, None)
    return module


class RunpodEndpointToolTests(unittest.TestCase):
    def test_validate_pinned_image_accepts_matching_sha_tag(self):
        tool = load_tool()

        image = tool.validate_pinned_image("styletts2_ua", "drgrandpa/styletts2-ua:sha-e706b9e")

        self.assertEqual(image, "drgrandpa/styletts2-ua:sha-e706b9e")

    def test_validate_pinned_image_rejects_latest(self):
        tool = load_tool()

        with self.assertRaisesRegex(ValueError, "latest"):
            tool.validate_pinned_image("styletts2_ua", "drgrandpa/styletts2-ua:latest")

    def test_validate_pinned_image_rejects_wrong_repo(self):
        tool = load_tool()

        with self.assertRaisesRegex(ValueError, "drgrandpa/styletts2-ua"):
            tool.validate_pinned_image("styletts2_ua", "drgrandpa/whisper-worker:sha-e706b9e")

    def test_validate_pinned_image_rejects_non_sha_tag(self):
        tool = load_tool()

        with self.assertRaisesRegex(ValueError, "sha-"):
            tool.validate_pinned_image("whisper_asr", "drgrandpa/whisper-worker:v1.2.3")

    def test_validate_pinned_image_accepts_separate_worker(self):
        tool = load_tool()

        image = tool.validate_pinned_image("separate_audio", "drgrandpa/separate-worker:sha-e706b9e")

        self.assertEqual(image, "drgrandpa/separate-worker:sha-e706b9e")

    def test_load_key_prefers_environment(self):
        tool = load_tool()

        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "from-env"}):
            self.assertEqual(tool.load_key(), "from-env")

    def test_load_key_reads_explicit_env_file(self):
        tool = load_tool()

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("RUNPOD_API_KEY=from-file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(tool.load_key(str(env_path)), "from-file")

    def test_main_rejects_latest_before_loading_api_key(self):
        tool = load_tool()

        with (
            mock.patch.object(tool, "load_key", side_effect=AssertionError("load_key should not run")),
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            tool.main(["create", "--worker", "styletts2_ua", "--image", "drgrandpa/styletts2-ua:latest"])


if __name__ == "__main__":
    unittest.main()
