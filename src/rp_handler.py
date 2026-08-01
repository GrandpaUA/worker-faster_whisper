import sys
import os

print("=" * 50, flush=True)
print("🚀 CONTAINER STARTING", flush=True)
print(f"   Python: {sys.version}", flush=True)
print(f"   Executable: {sys.executable}", flush=True)
print(f"   CWD: {os.getcwd()}", flush=True)
print(f"   Files in /: {os.listdir('/')}", flush=True)
print("=" * 50, flush=True)

print("📦 Importing base64, tempfile...", flush=True)
import base64
import subprocess
import tempfile
import time
print("   ✅ base64, tempfile OK", flush=True)

print("📦 Importing rp_schema...", flush=True)
from rp_schema import INPUT_VALIDATIONS
print("   ✅ rp_schema OK", flush=True)

print("📦 Importing runpod.serverless.utils...", flush=True)
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
print("   ✅ runpod.serverless.utils OK", flush=True)

print("📦 Importing runpod...", flush=True)
import runpod
print(f"   ✅ runpod OK (version: {runpod.__version__ if hasattr(runpod, '__version__') else 'unknown'})", flush=True)

print("📦 Importing predict...", flush=True)
import predict
print("   ✅ predict OK", flush=True)

print("🔧 Creating Predictor...", flush=True)
MODEL = predict.Predictor()
print("   ✅ Predictor created", flush=True)

print("🔧 Running setup()...", flush=True)
MODEL.setup()
print("   ✅ setup() done", flush=True)

# CUDA check
try:
    from runpod.serverless.utils import rp_cuda
    print(f"🎮 CUDA available: {rp_cuda.is_available()}", flush=True)
except Exception as e:
    print(f"⚠️ CUDA check failed: {e}", flush=True)


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],

            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],

            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"]
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


@rp_debugger.FunctionTimer
def run_demucs_job(job):
    job_input = job['input']

    input_validation = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in input_validation:
        return {"error": input_validation['errors']}
    job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
    else:
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    model = job_input.get('demucs_model', 'htdemucs_ft')

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        cuda_available = False
        gpu_name = None

    outdir = tempfile.mkdtemp()

    # --two-stems vocals → лише vocals.mp3 + no_vocals.mp3 (сума решти stem'ів).
    # --mp3-bitrate 64: stems ~8МБ base64 кожен — обидва пролізають у ліміт
    # відповіді RunPod (~20МБ); 320k (дефолт) дає ~74МБ і output занулюється.
    # GPU підхоплюється автоматично, якщо torch бачить CUDA.
    cmd = [
        sys.executable, '-m', 'demucs',
        '--mp3', '--mp3-bitrate', '64', '--two-stems', 'vocals',
        '-n', model, '-o', outdir, audio_input,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    separation_sec = round(time.time() - t0, 2)

    if proc.returncode != 0:
        return {
            'error': f'demucs failed (exit {proc.returncode})',
            'separation_sec': separation_sec,
            'model': model,
            'cuda_available': cuda_available,
            'stderr': (proc.stderr or '')[-4000:],
        }

    stem_name = os.path.splitext(os.path.basename(audio_input))[0]
    stem_dir = os.path.join(outdir, model, stem_name)

    result = {
        'separation_sec': separation_sec,
        'model': model,
        'cuda_available': cuda_available,
        'gpu_name': gpu_name,
    }
    # return_stems=False → лише метадані (розміри, час, GPU). Повні base64 stem'и
    # перевищують ліміт відповіді RunPod (~20 МБ) і output занулюється.
    return_stems = job_input.get('return_stems', True)
    for stem in ('no_vocals', 'vocals'):
        path = os.path.join(stem_dir, f'{stem}.mp3')
        if os.path.exists(path):
            result[f'{stem}_bytes'] = os.path.getsize(path)
            if return_stems:
                with open(path, 'rb') as f:
                    result[f'{stem}_base64'] = base64.b64encode(f.read()).decode()
        else:
            result[f'{stem}_error'] = f'missing {path}'

    return result


def handler(job):
    task = (job.get('input') or {}).get('task', 'transcribe')
    if task == 'separate':
        return run_demucs_job(job)
    return run_whisper_job(job)


print("🌐 Starting runpod.serverless...", flush=True)
runpod.serverless.start({"handler": handler})
print("✅ runpod.serverless started — worker ready!", flush=True)

# GHCR build trigger
