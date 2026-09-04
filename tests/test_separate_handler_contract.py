import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class _CompletedProcess:
    returncode = 0
    stderr = ""


def _identity_timer(fn):
    return fn


def _validate(job_input, schema):
    errors = []
    validated = {}
    for name, rules in schema.items():
        value = job_input.get(name, rules.get("default"))
        expected = rules["type"]
        if value is not None:
            is_valid = isinstance(value, expected)
            if expected is float and isinstance(value, int):
                is_valid = True
            if not is_valid:
                errors.append(f"{name} must be {expected.__name__}")
        validated[name] = value
    if errors:
        return {"errors": errors}
    return {"validated_input": validated}


class _SeparateHandlerLoader:
    def __init__(self):
        self.start_calls = []
        self.download_calls = []
        self._saved_modules = {}
        self.module = None

    def __enter__(self):
        self._install_stubs()
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        spec = importlib.util.spec_from_file_location("separate_handler_under_test", SRC / "separate_handler.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        with mock.patch("sys.stdout", io.StringIO()):
            spec.loader.exec_module(module)
        self.module = module
        return self

    def __exit__(self, *_args):
        sys.modules.pop("separate_handler_under_test", None)
        for name in self._stub_names:
            sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)

    def _remember(self, name):
        if name in sys.modules and name not in self._saved_modules:
            self._saved_modules[name] = sys.modules[name]

    def _install(self, name, module):
        self._remember(name)
        sys.modules[name] = module

    def _install_stubs(self):
        self._stub_names = [
            "separate",
            "runpod",
            "runpod.serverless",
            "runpod.serverless.utils",
            "runpod.serverless.utils.rp_validator",
            "torch",
        ]

        runpod_module = types.ModuleType("runpod")
        runpod_module.__version__ = "test"

        serverless_module = types.ModuleType("runpod.serverless")

        def start(config):
            self.start_calls.append(config)

        serverless_module.start = start
        runpod_module.serverless = serverless_module

        utils_module = types.ModuleType("runpod.serverless.utils")

        def download_files_from_urls(job_id, urls):
            self.download_calls.append((job_id, urls))
            return [f"/tmp/{job_id}.wav"]

        utils_module.download_files_from_urls = download_files_from_urls
        utils_module.rp_debugger = types.SimpleNamespace(FunctionTimer=_identity_timer)

        validator_module = types.ModuleType("runpod.serverless.utils.rp_validator")
        validator_module.validate = _validate

        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda _index: "fake-gpu",
        )

        self._install("runpod", runpod_module)
        self._install("runpod.serverless", serverless_module)
        self._install("runpod.serverless.utils", utils_module)
        self._install("runpod.serverless.utils.rp_validator", validator_module)
        self._install("torch", torch_module)


class SeparateHandlerContractTests(unittest.TestCase):
    def test_import_does_not_load_whisper_predictor_or_start_serverless(self):
        with _SeparateHandlerLoader() as loaded:
            self.assertEqual(loaded.start_calls, [])
            self.assertNotIn("predict", sys.modules)

    def test_rejects_transcription_task(self):
        with _SeparateHandlerLoader() as loaded:
            result = loaded.module.handler(
                {
                    "id": "job-1",
                    "input": {
                        "task": "transcribe",
                        "audio_base64": "ZmFrZQ==",
                    },
                }
            )

            self.assertEqual(result, {"error": "This endpoint supports only task='separate'"})

    def test_unknown_separation_engine_returns_error(self):
        with _SeparateHandlerLoader() as loaded:
            result = loaded.module.handler(
                {
                    "id": "job-1",
                    "input": {
                        "task": "separate",
                        "engine": "spleeter",
                        "audio_base64": "ZmFrZQ==",
                    },
                }
            )

            self.assertEqual(result, {"error": "Unknown engine 'spleeter', expected 'demucs' or 'roformer'"})

    def test_demucs_metadata_mode_omits_base64_stems(self):
        with _SeparateHandlerLoader() as loaded:
            separate = sys.modules["separate"]
            with (
                mock.patch.object(separate.subprocess, "run", return_value=_CompletedProcess()),
                mock.patch.object(separate.tempfile, "mkdtemp", return_value="/tmp/out"),
                mock.patch.object(separate.os.path, "exists", return_value=True),
                mock.patch.object(separate.os.path, "getsize", return_value=1234),
            ):
                result = loaded.module.handler(
                    {
                        "id": "job-1",
                        "input": {
                            "task": "separate",
                            "engine": "demucs",
                            "audio": "https://example.test/audio.wav",
                            "return_stems": False,
                        },
                    }
                )

            self.assertEqual(result["model"], "htdemucs_ft")
            self.assertEqual(result["cuda_available"], False)
            self.assertEqual(result["vocals_bytes"], 1234)
            self.assertEqual(result["no_vocals_bytes"], 1234)
            self.assertNotIn("vocals_base64", result)
            self.assertNotIn("no_vocals_base64", result)


if __name__ == "__main__":
    unittest.main()
