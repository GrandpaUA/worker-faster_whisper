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

    def test_info_uses_runpod_rest_v2_endpoint_lookup(self):
        tool = load_tool()

        client = mock.Mock()
        client.http.return_value = {"id": "endpoint-1", "image": "drgrandpa/whisper-worker:sha-8968539"}

        with mock.patch("sys.stdout"):
            tool.cmd_info(client, "endpoint-1")

        client.http.assert_called_once_with(
            "https://api.runpod.io/v2/serverless/endpoint-1",
            method="GET",
            exit_on_error=False,
        )
        client.graphql.assert_not_called()

    def test_info_falls_back_to_graphql_endpoint_list_on_rest_404(self):
        tool = load_tool()

        client = mock.Mock()
        client.http.side_effect = tool.RunpodHTTPError(404, "https://api.runpod.io/v2/serverless/endpoint-1", "{}")
        client.graphql.return_value = {
            "myself": {
                "endpoints": [
                    {
                        "id": "endpoint-1",
                        "name": "legacy",
                        "gpuIds": "AMPERE_16",
                        "templateId": "template-1",
                    }
                ]
            }
        }

        with mock.patch("sys.stdout"):
            tool.cmd_info(client, "endpoint-1")

        client.graphql.assert_called_once()

    def test_info_reports_visible_endpoints_when_id_is_not_found(self):
        tool = load_tool()

        client = mock.Mock()
        client.http.side_effect = tool.RunpodHTTPError(404, "https://api.runpod.io/v2/serverless/missing", "{}")
        client.graphql.return_value = {
            "myself": {
                "endpoints": [
                    {"id": "endpoint-1", "name": "visible"},
                ]
            }
        }

        with self.assertRaisesRegex(SystemExit, "endpoint-1 \\(visible\\)"):
            tool.cmd_info(client, "missing")


if __name__ == "__main__":
    unittest.main()
