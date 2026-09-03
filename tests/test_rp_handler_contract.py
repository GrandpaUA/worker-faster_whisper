import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class _Timer:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


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


class _FakePredictor:
    instances = []

    def __init__(self):
        self.setup_calls = 0
        self.predict_calls = []
        self.__class__.instances.append(self)

    def setup(self):
        self.setup_calls += 1

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        result = {
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 1.2,
                    "text": " Hello.",
                    "tokens": [50364],
                    "temperature": 0.0,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01,
                }
            ],
            "detected_language": kwargs["language"] or "en",
            "transcription": "Hello.",
            "translation": None,
            "device": "cuda",
            "model": kwargs["model_name"],
        }
        if kwargs["word_timestamps"]:
            result["word_timestamps"] = [{"word": "Hello", "start": 0.0, "end": 1.0}]
        return result


class _HandlerLoader:
    def __init__(self):
        self.start_calls = []
        self.download_calls = []
        self.cleanup_calls = []
        self._saved_modules = {}
        self.module = None

    def __enter__(self):
        self._install_stubs()
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        spec = importlib.util.spec_from_file_location("rp_handler_under_test", SRC / "rp_handler.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        with mock.patch("sys.stdout", io.StringIO()):
            spec.loader.exec_module(module)
        self.module = module
        return self

    def __exit__(self, *_args):
        sys.modules.pop("rp_handler_under_test", None)
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
        _FakePredictor.instances = []
        self._stub_names = [
            "predict",
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
        utils_module.rp_cleanup = types.SimpleNamespace(clean=lambda keys: self.cleanup_calls.append(keys))
        utils_module.rp_debugger = types.SimpleNamespace(FunctionTimer=_identity_timer, LineTimer=_Timer)
        utils_module.rp_cuda = types.SimpleNamespace(is_available=lambda: True)

        validator_module = types.ModuleType("runpod.serverless.utils.rp_validator")
        validator_module.validate = _validate

        predict_module = types.ModuleType("predict")
        predict_module.Predictor = _FakePredictor

        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda _index: "fake-gpu",
        )

        self._install("runpod", runpod_module)
        self._install("runpod.serverless", serverless_module)
        self._install("runpod.serverless.utils", utils_module)
        self._install("runpod.serverless.utils.rp_validator", validator_module)
        self._install("predict", predict_module)
        self._install("torch", torch_module)


class RunpodHandlerContractTests(unittest.TestCase):
    def test_import_sets_up_model_without_starting_serverless(self):
        with _HandlerLoader() as loaded:
            self.assertEqual(loaded.start_calls, [])
            self.assertEqual(len(_FakePredictor.instances), 1)
            self.assertEqual(_FakePredictor.instances[0].setup_calls, 1)

    def test_transcription_contract_and_forwarded_options(self):
        with _HandlerLoader() as loaded:
            result = loaded.module.handler(
                {
                    "id": "job-1",
                    "input": {
                        "audio": "https://example.test/audio.wav",
                        "model": "large-v2",
                        "language": "en",
                        "word_timestamps": True,
                    },
                }
            )

            self.assertIn("segments", result)
            self.assertEqual(result["detected_language"], "en")
            self.assertEqual(result["model"], "large-v2")
            self.assertEqual(result["device"], "cuda")
            self.assertIn("word_timestamps", result)
            self.assertEqual(loaded.download_calls, [("job-1", ["https://example.test/audio.wav"])])
            self.assertEqual(loaded.cleanup_calls, [["input_objects"]])

            call = _FakePredictor.instances[0].predict_calls[0]
            self.assertEqual(call["audio"], "/tmp/job-1.wav")
            self.assertEqual(call["model_name"], "large-v2")
            self.assertEqual(call["language"], "en")
            self.assertEqual(call["transcription"], "plain_text")
            self.assertFalse(call["translate"])
            self.assertTrue(call["condition_on_previous_text"])
            self.assertTrue(call["word_timestamps"])

    def test_omitted_model_defaults_to_large_v2(self):
        with _HandlerLoader() as loaded:
            result = loaded.module.handler(
                {
                    "id": "job-1",
                    "input": {
                        "audio": "https://example.test/audio.wav",
                    },
                }
            )

            self.assertEqual(result["model"], "large-v2")
            call = _FakePredictor.instances[0].predict_calls[0]
            self.assertEqual(call["model_name"], "large-v2")

    def test_rejects_missing_or_duplicate_audio_sources(self):
        with _HandlerLoader() as loaded:
            missing = loaded.module.handler({"id": "job-1", "input": {"model": "large-v2"}})
            duplicate = loaded.module.handler(
                {
                    "id": "job-2",
                    "input": {
                        "audio": "https://example.test/audio.wav",
                        "audio_base64": "ZmFrZQ==",
                    },
                }
            )

            self.assertEqual(missing, {"error": "Must provide either audio or audio_base64"})
            self.assertEqual(duplicate, {"error": "Must provide either audio or audio_base64, not both"})

    def test_validation_errors_are_returned_loudly(self):
        with _HandlerLoader() as loaded:
            result = loaded.module.handler({"id": "job-1", "input": {"audio": 123}})

            self.assertIn("error", result)
            self.assertIn("audio must be str", result["error"])

    def test_unknown_separation_engine_returns_error(self):
        with _HandlerLoader() as loaded:
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
        with _HandlerLoader() as loaded:
            module = loaded.module
            with (
                mock.patch.object(module.subprocess, "run", return_value=_CompletedProcess()),
                mock.patch.object(module.tempfile, "mkdtemp", return_value="/tmp/out"),
                mock.patch.object(module.os.path, "exists", return_value=True),
                mock.patch.object(module.os.path, "getsize", return_value=1234),
            ):
                result = module.handler(
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
