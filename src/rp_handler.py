"""
rp_handler.py — simple HTTP server compatible with RunPod aiapi runtime.
Replaces runpod.serverless.start() which is incompatible with the new runtime.
"""
import os
import json
import base64
import tempfile
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

import predict

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        print("[handler] Loading model...")
        MODEL = predict.Predictor()
        MODEL.setup()
        print("[handler] Model loaded.")
    return MODEL


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            self._json_response(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
            job_input = data.get("input", data)

            # basic_test or missing audio — return empty result
            if not job_input.get("audio") and not job_input.get("audio_base64"):
                self._json_response(200, {
                    "segments": [],
                    "detected_language": None,
                    "transcription": "",
                })
                return

            # Resolve audio file
            if job_input.get("audio_base64"):
                audio_input = base64_to_tempfile(job_input["audio_base64"])
            else:
                from runpod.serverless.utils import download_files_from_urls
                audio_input = download_files_from_urls(
                    data.get("id", "job"), [job_input["audio"]]
                )[0]

            model = get_model()
            result = model.predict(
                audio=audio_input,
                model_name=job_input.get("model", "small"),
                transcription=job_input.get("transcription", "plain_text"),
                translation=job_input.get("translation", "plain_text"),
                translate=job_input.get("translate", False),
                language=job_input.get("language", None),
                temperature=job_input.get("temperature", 0),
                best_of=job_input.get("best_of", 5),
                beam_size=job_input.get("beam_size", 5),
                patience=job_input.get("patience", None),
                length_penalty=job_input.get("length_penalty", None),
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input.get("initial_prompt", None),
                condition_on_previous_text=job_input.get("condition_on_previous_text", True),
                temperature_increment_on_fallback=job_input.get("temperature_increment_on_fallback", 0.2),
                compression_ratio_threshold=job_input.get("compression_ratio_threshold", 2.4),
                logprob_threshold=job_input.get("logprob_threshold", -1.0),
                no_speech_threshold=job_input.get("no_speech_threshold", 0.6),
                enable_vad=job_input.get("enable_vad", False),
                word_timestamps=job_input.get("word_timestamps", False),
            )

            self._json_response(200, result)

        except Exception as e:
            traceback.print_exc()
            self._json_response(500, {"error": str(e)})

    def _json_response(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[handler] {args[0]}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[handler] Listening on 0.0.0.0:{port}")
    server.serve_forever()
