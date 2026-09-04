"""RunPod endpoint deploy/test tool for YouTube Voice Translator workers.

Commands:
  python tools/runpod_endpoint.py gpus
  python tools/runpod_endpoint.py create --worker styletts2_ua --image drgrandpa/styletts2-ua:sha-abcdef1
  python tools/runpod_endpoint.py create --worker whisper_asr --image drgrandpa/whisper-worker:sha-abcdef1
  python tools/runpod_endpoint.py create --worker separate_audio --image drgrandpa/separate-worker:sha-abcdef1
  python tools/runpod_endpoint.py info <endpoint_id>
  python tools/runpod_endpoint.py run-styletts2 <endpoint_id> [--text "..."] [--voice "..."] [--out file.wav]
  python tools/runpod_endpoint.py fetch-styletts2 <endpoint_id> <job_id> [out.wav]

RUNPOD_API_KEY is read from the environment first, then from .env. When the tool
runs from this worker repo, it also checks the sibling youtube-translator
checkout at ../subtitres/.env.

Production deploy rule: create requires a pinned sha tag, for example
drgrandpa/styletts2-ua:sha-e706b9e. Mutable tags such as latest are rejected.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import wave


ROOT = Path(__file__).resolve().parents[1]
GRAPHQL = "https://api.runpod.io/graphql"
REST = "https://api.runpod.ai"
HOURLY_USD = 0.58  # RTX 4000 Ada

# GPU для endpoint'а - тільки точна назва карти з gpuTypes, без pool aliases.
GPU_IDS_RTX4000_ADA = "NVIDIA RTX 4000 Ada Generation"
PINNED_SHA_RE = re.compile(r"^sha-[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class WorkerSpec:
    image_repo: str
    endpoint_name: str
    template_name: str
    disk_gb: int


WORKERS = {
    "whisper_asr": WorkerSpec(
        image_repo="drgrandpa/whisper-worker",
        endpoint_name="whisper-asr",
        template_name="whisper-asr-tpl",
        disk_gb=30,
    ),
    "separate_audio": WorkerSpec(
        image_repo="drgrandpa/separate-worker",
        endpoint_name="separate-audio",
        template_name="separate-audio-tpl",
        disk_gb=30,
    ),
    "styletts2_ua": WorkerSpec(
        image_repo="drgrandpa/styletts2-ua",
        endpoint_name="styletts2-ua-eval",
        template_name="styletts2-ua-eval-tpl",
        disk_gb=20,
    ),
}


def candidate_env_paths(explicit: str | None = None) -> list[Path]:
    paths = []
    if explicit:
        paths.append(Path(explicit))
    env_path = os.getenv("RUNPOD_ENV_PATH")
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.cwd() / ".env")
    paths.append(ROOT / ".env")
    paths.append(ROOT.parent / "subtitres" / ".env")
    return paths


def load_key(explicit_env_path: str | None = None) -> str:
    env_key = os.getenv("RUNPOD_API_KEY")
    if env_key:
        return env_key

    checked = []
    for path in candidate_env_paths(explicit_env_path):
        checked.append(str(path))
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("RUNPOD_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value

    raise SystemExit("RUNPOD_API_KEY not found. Checked: " + ", ".join(checked))


def split_image(image: str) -> tuple[str, str]:
    if ":" not in image:
        raise ValueError("image must include an explicit tag, e.g. repo/name:sha-abcdef1")
    repo, tag = image.rsplit(":", 1)
    if not repo or not tag:
        raise ValueError("image must include both repository and tag")
    return repo, tag


def validate_pinned_image(worker: str, image: str) -> str:
    if worker not in WORKERS:
        raise ValueError(f"unknown worker {worker!r}")

    repo, tag = split_image(image)
    expected_repo = WORKERS[worker].image_repo
    if repo != expected_repo:
        raise ValueError(f"{worker} image must use {expected_repo}, got {repo}")
    if tag == "latest":
        raise ValueError("latest is mutable and is not allowed for endpoint deploy")
    if not PINNED_SHA_RE.match(tag):
        raise ValueError("endpoint deploy requires sha-* image tag, e.g. sha-e706b9e")
    return image


def gql_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


class RunpodClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def http(self, url: str, payload=None, method: str | None = None):
        # Cloudflare RunPod банить Python-urllib UA (error 1010) - даємо нейтральний.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "runpod-python/1.9.0",
        }
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise SystemExit(f"HTTP {exc.code} for {url}:\n{body[:2000]}") from exc

    def graphql(self, query: str):
        result = self.http(GRAPHQL, {"query": query})
        if result.get("errors"):
            errors = json.dumps(result["errors"], indent=2, ensure_ascii=False)
            raise SystemExit(f"GraphQL errors:\n{errors}")
        return result["data"]


def cmd_gpus(client: RunpodClient) -> None:
    data = client.graphql("query { gpuTypes { id displayName secureCloud } }")
    for gpu in sorted(data["gpuTypes"], key=lambda item: item["displayName"]):
        marker = " <== RTX 4000 Ada?" if "ADA" in gpu["id"].upper() else ""
        print(f"{gpu['id']:24s} {gpu['displayName']}{marker}")


def cmd_create(client: RunpodClient, args: argparse.Namespace) -> None:
    spec = WORKERS[args.worker]
    image = validate_pinned_image(args.worker, args.image)
    template_name = args.template_name or spec.template_name
    endpoint_name = args.endpoint_name or spec.endpoint_name
    disk_gb = args.disk_gb if args.disk_gb is not None else spec.disk_gb

    print(f"1/2 saveTemplate (worker {args.worker}, image {image}, disk {disk_gb}GB)...")
    template = client.graphql(f"""
    mutation {{
        saveTemplate(input: {{
            name: {gql_str(template_name)},
            imageName: {gql_str(image)},
            dockerArgs: "",
            containerDiskInGb: {disk_gb},
            volumeInGb: 0,
            ports: "",
            env: [],
            isServerless: true,
            containerRegistryAuthId: "",
            startSsh: true,
            isPublic: false,
            readme: ""
        }}) {{ id name imageName containerDiskInGb isServerless }}
    }}""")["saveTemplate"]
    print(json.dumps(template, indent=2, ensure_ascii=False))

    print(f"2/2 saveEndpoint (gpuIds={args.gpu_ids}, workers {args.workers_min}..{args.workers_max})...")
    endpoint = client.graphql(f"""
    mutation {{
        saveEndpoint(input: {{
            name: {gql_str(endpoint_name)},
            templateId: {gql_str(template["id"])},
            gpuIds: {gql_str(args.gpu_ids)},
            networkVolumeId: "",
            locations: "",
            idleTimeout: {args.idle_timeout},
            scalerType: "QUEUE_DELAY",
            scalerValue: {args.scaler_value},
            workersMin: {args.workers_min},
            workersMax: {args.workers_max}
        }}) {{ id name templateId gpuIds locations workersMin workersMax }}
    }}""")["saveEndpoint"]
    print(json.dumps(endpoint, indent=2, ensure_ascii=False))
    print(f"\nENDPOINT_ID={endpoint['id']}")
    print("Manual check in RunPod console: FlashBoot is not controlled by saveEndpoint.")


def cmd_info(client: RunpodClient, endpoint_id: str) -> None:
    data = client.graphql(f"""
    query {{
        endpoint(id: {gql_str(endpoint_id)}) {{
            id name gpuIds locations workersMin workersMax idleTimeout
            template {{ id name imageName }}
        }}
    }}""")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def check_wav(path: str) -> bool:
    with wave.open(path, "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        print(
            f"wave: channels={wav.getnchannels()} sampwidth={wav.getsampwidth()} "
            f"rate={wav.getframerate()} frames={wav.getnframes()} duration={duration:.2f}s"
        )
        ok = (
            wav.getnchannels() == 1
            and wav.getsampwidth() == 2
            and wav.getframerate() == 22050
            and duration > 2.0
        )
    print("WAV CHECK:", "OK" if ok else "MISMATCH (expected mono 16-bit 22050 Hz >2s)")
    return ok


def cmd_run_styletts2(
    client: RunpodClient,
    endpoint_id: str,
    text: str,
    voice: str | None,
    out: str,
) -> None:
    payload = {"input": {"text": text}}
    if voice:
        payload["input"]["voice"] = voice
    print(f"POST /v2/{endpoint_id}/run text={text!r} voice={voice!r}")
    job = client.http(f"{REST}/v2/{endpoint_id}/run", payload)
    job_id = job["id"]
    print(f"job_id={job_id} status={job.get('status')}")

    start = time.time()
    while True:
        time.sleep(5)
        status = client.http(f"{REST}/v2/{endpoint_id}/status/{job_id}", method="GET")
        state = status.get("status")
        print(f"[{time.time() - start:6.1f}s] {state}", flush=True)
        if state == "COMPLETED":
            break
        if state in ("FAILED", "CANCELLED"):
            print(json.dumps(status, indent=2, ensure_ascii=False)[:3000])
            raise SystemExit(f"job {state}")
        if time.time() - start > 600:
            raise SystemExit("timeout 600s")

    save_styletts2_result(status, out)


def save_styletts2_result(status: dict, out: str) -> None:
    print(f"delayTime={status.get('delayTime')}ms executionTime={status.get('executionTime')}ms")
    milliseconds = (status.get("delayTime") or 0) + (status.get("executionTime") or 0)
    print(f"cost ~ ${milliseconds / 3600000 * HOURLY_USD:.6f} (${HOURLY_USD}/hr)")
    print(f"workerId={status.get('workerId')}")

    output = status.get("output")
    if output is None:
        raise SystemExit("output=null - response >20MB or worker died")
    meta = {key: value for key, value in output.items() if key != "audio_base64"}
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if "error" in output:
        raise SystemExit("worker returned error")
    audio_base64 = output.get("audio_base64")
    if not audio_base64:
        raise SystemExit("audio_base64 is empty")
    raw = base64.b64decode(audio_base64)
    with open(out, "wb") as audio:
        audio.write(raw)
    print(f"audio: {len(raw)} bytes -> {out}")
    check_wav(out)


def cmd_fetch_styletts2(client: RunpodClient, endpoint_id: str, job_id: str, out: str) -> None:
    status = client.http(f"{REST}/v2/{endpoint_id}/status/{job_id}", method="GET")
    print(f"status={status.get('status')}")
    if status.get("status") != "COMPLETED":
        raise SystemExit(f"job is not COMPLETED: {status.get('status')}")
    save_styletts2_result(status, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", help="Path to .env with RUNPOD_API_KEY")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gpus", help="List RunPod GPU types")

    create = subparsers.add_parser("create", help="Create template and endpoint with pinned image")
    create.add_argument("--worker", choices=sorted(WORKERS), default="styletts2_ua")
    create.add_argument("--image", required=True, help="Pinned image, e.g. drgrandpa/styletts2-ua:sha-e706b9e")
    create.add_argument("--gpu-ids", default=GPU_IDS_RTX4000_ADA)
    create.add_argument("--template-name")
    create.add_argument("--endpoint-name")
    create.add_argument("--disk-gb", type=int)
    create.add_argument("--workers-min", type=int, default=0)
    create.add_argument("--workers-max", type=int, default=1)
    create.add_argument("--idle-timeout", type=int, default=5)
    create.add_argument("--scaler-value", type=int, default=4)

    info = subparsers.add_parser("info", help="Show endpoint template and settings")
    info.add_argument("endpoint_id")

    run_tts = subparsers.add_parser("run-styletts2", help="Run StyleTTS2 smoke job and save WAV")
    run_tts.add_argument("endpoint_id")
    run_tts.add_argument("--text", default="Привіт! Це тест українського синтезу мовлення.")
    run_tts.add_argument("--voice")
    run_tts.add_argument("--out", default="styletts2_test.wav")

    fetch_tts = subparsers.add_parser("fetch-styletts2", help="Fetch completed StyleTTS2 job and save WAV")
    fetch_tts.add_argument("endpoint_id")
    fetch_tts.add_argument("job_id")
    fetch_tts.add_argument("out", nargs="?", default="styletts2_test.wav")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        try:
            validate_pinned_image(args.worker, args.image)
        except ValueError as exc:
            parser.error(str(exc))

    client = RunpodClient(load_key(args.env))

    if args.command == "gpus":
        cmd_gpus(client)
    elif args.command == "create":
        cmd_create(client, args)
    elif args.command == "info":
        cmd_info(client, args.endpoint_id)
    elif args.command == "run-styletts2":
        cmd_run_styletts2(client, args.endpoint_id, args.text, args.voice, args.out)
    elif args.command == "fetch-styletts2":
        cmd_fetch_styletts2(client, args.endpoint_id, args.job_id, args.out)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
