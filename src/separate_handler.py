import os
import sys

print("=" * 50, flush=True)
print("SEPARATION CONTAINER STARTING", flush=True)
print(f"   Python: {sys.version}", flush=True)
print(f"   Executable: {sys.executable}", flush=True)
print(f"   CWD: {os.getcwd()}", flush=True)
print(f"   Files in /: {os.listdir('/')}", flush=True)
print("=" * 50, flush=True)

print("Importing runpod...", flush=True)
import runpod

print(f"   runpod OK (version: {runpod.__version__ if hasattr(runpod, '__version__') else 'unknown'})", flush=True)

print("Importing separation job handlers...", flush=True)
from separate import run_demucs_job, run_roformer_job

print("   separation handlers OK", flush=True)

try:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as exc:
    print(f"CUDA check failed: {exc}", flush=True)


def handler(job):
    job_input = job.get("input") or {}
    task = job_input.get("task", "separate")
    if task != "separate":
        return {"error": "This endpoint supports only task='separate'"}

    engine = job_input.get("engine", "demucs")
    if engine == "roformer":
        return run_roformer_job(job)
    if engine != "demucs":
        return {"error": f"Unknown engine '{engine}', expected 'demucs' or 'roformer'"}
    return run_demucs_job(job)


if __name__ == "__main__":
    print("Starting runpod.serverless...", flush=True)
    runpod.serverless.start({"handler": handler})
    print("runpod.serverless started - worker ready!", flush=True)
