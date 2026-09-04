import base64
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_debugger
from runpod.serverless.utils.rp_validator import validate


DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
ROFORMER_MODEL_DIR = "/tmp/audio-separator-models/"
DEFAULT_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


def validate_audio_input(job):
    job_input = job["input"]

    input_validation = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in input_validation:
        return None, {"error": input_validation["errors"]}
    job_input = input_validation["validated_input"]

    if not job_input.get("audio", False) and not job_input.get("audio_base64", False):
        return None, {"error": "Must provide either audio or audio_base64"}

    if job_input.get("audio", False) and job_input.get("audio_base64", False):
        return None, {"error": "Must provide either audio or audio_base64, not both"}

    if job_input.get("audio", False):
        audio_input = download_files_from_urls(job["id"], [job_input["audio"]])[0]
    else:
        audio_input = base64_to_tempfile(job_input["audio_base64"])

    return (job_input, audio_input), None


def get_cuda_meta():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        cuda_available = False
        gpu_name = None
    return cuda_available, gpu_name


@rp_debugger.FunctionTimer
def run_demucs_job(job):
    validated, error = validate_audio_input(job)
    if error:
        return error
    job_input, audio_input = validated

    model = job_input.get("demucs_model", DEFAULT_DEMUCS_MODEL)
    cuda_available, gpu_name = get_cuda_meta()
    outdir = tempfile.mkdtemp()

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--mp3",
        "--mp3-bitrate",
        "64",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-o",
        outdir,
        audio_input,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    separation_sec = round(time.time() - t0, 2)

    if proc.returncode != 0:
        return {
            "error": f"demucs failed (exit {proc.returncode})",
            "separation_sec": separation_sec,
            "model": model,
            "cuda_available": cuda_available,
            "stderr": (proc.stderr or "")[-4000:],
        }

    stem_name = os.path.splitext(os.path.basename(audio_input))[0]
    stem_dir = os.path.join(outdir, model, stem_name)

    result = {
        "separation_sec": separation_sec,
        "model": model,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }
    return_stems = job_input.get("return_stems", True)
    for stem in ("no_vocals", "vocals"):
        path = os.path.join(stem_dir, f"{stem}.mp3")
        if os.path.exists(path):
            result[f"{stem}_bytes"] = os.path.getsize(path)
            if return_stems:
                with open(path, "rb") as f:
                    result[f"{stem}_base64"] = base64.b64encode(f.read()).decode()
        else:
            result[f"{stem}_error"] = f"missing {path}"

    return result


@rp_debugger.FunctionTimer
def run_roformer_job(job):
    validated, error = validate_audio_input(job)
    if error:
        return error
    job_input, audio_input = validated

    model = job_input.get("roformer_model", DEFAULT_ROFORMER_MODEL)
    cuda_available, gpu_name = get_cuda_meta()
    outdir = tempfile.mkdtemp()

    t0 = time.time()
    try:
        from audio_separator.separator import Separator

        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=ROFORMER_MODEL_DIR,
            output_dir=outdir,
            output_format="WAV",
        )
        torch_device = str(getattr(separator, "torch_device", None) or "unknown")
        onnx_provider = getattr(separator, "onnx_execution_provider", None)
        if not cuda_available:
            print("WARNING: roformer torch.cuda.is_available()=False; using CPU", flush=True)
        elif torch_device != "cuda":
            print(f"WARNING: roformer CUDA is available but Separator selected {torch_device}", flush=True)

        separator.load_model(model_filename=model)
        output_files = separator.separate(audio_input, {"Vocals": "vocals", "Instrumental": "no_vocals"})
    except Exception:
        return {
            "error": "audio-separator failed",
            "model": model,
            "engine": "roformer",
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "stderr": traceback.format_exc()[-4000:],
        }
    separation_sec = round(time.time() - t0, 2)

    result = {
        "separation_sec": separation_sec,
        "model": model,
        "engine": "roformer",
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "torch_device": torch_device,
        "onnx_execution_provider": onnx_provider,
    }

    return_stems = job_input.get("return_stems", True)
    for stem in ("no_vocals", "vocals"):
        wav_path = os.path.join(outdir, f"{stem}.wav")
        mp3_path = os.path.join(outdir, f"{stem}.mp3")
        if not os.path.exists(wav_path):
            result[f"{stem}_error"] = f"missing {wav_path} (separator output: {output_files})"
            continue
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ac", "2", "-b:a", "64k", mp3_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            result[f"{stem}_error"] = f'ffmpeg failed (exit {proc.returncode}): {(proc.stderr or "")[-2000:]}'
            continue
        result[f"{stem}_bytes"] = os.path.getsize(mp3_path)
        if return_stems:
            with open(mp3_path, "rb") as f:
                result[f"{stem}_base64"] = base64.b64encode(f.read()).decode()

    return result
