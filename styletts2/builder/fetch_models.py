"""Префетч моделей StyleTTS2 UA у образ при білді (runtime без скачувань).

Порядок: TTS-модель → голоси → verbalizer → stressifier (stanza 'uk') →
саніті-синтез на CPU. Саніті ловить несумісність версій/моделей на білді,
а не на холодному старті endpoint'а.
"""
import glob
import os
import re
import shutil
import unicodedata

import torch
from huggingface_hub import snapshot_download

MODEL_REPO = "patriotyk/styletts2_ukrainian_multispeaker_istftnet"
SPACE_REPO = "patriotyk/styletts2-ukrainian"
VOICES_DIR = "/voices"
SANITY_VOICE = "Марина Панас"


def fetch_tts_model():
    print(f"⬇️ Downloading TTS model {MODEL_REPO}...", flush=True)
    snapshot_download(MODEL_REPO)
    print("✅ TTS model cached (HF cache)", flush=True)


def fetch_voices():
    print(f"⬇️ Downloading voices from space {SPACE_REPO}...", flush=True)
    raw_dir = "/voices_raw"
    snapshot_download(
        SPACE_REPO,
        repo_type="space",
        allow_patterns=["voices/*.pt"],
        local_dir=raw_dir,
    )
    # local_dir зберігає структуру репо (voices/*.pt → raw/voices/*.pt) —
    # флетенимо у /voices, звідки їх читає handler.
    src = sorted(glob.glob(os.path.join(raw_dir, "voices", "*.pt")))
    if not src:
        src = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pt"), recursive=True))
    os.makedirs(VOICES_DIR, exist_ok=True)
    for path in src:
        shutil.move(path, os.path.join(VOICES_DIR, os.path.basename(path)))
    shutil.rmtree(raw_dir, ignore_errors=True)
    voices = sorted(glob.glob(os.path.join(VOICES_DIR, "*.pt")))
    names = [os.path.basename(v)[:-3] for v in voices]
    print(f"✅ {len(voices)} voices: {names}", flush=True)
    if not voices:
        raise RuntimeError("no voice .pt files downloaded")
    if not any(os.path.basename(v)[:-3] == SANITY_VOICE for v in voices):
        raise RuntimeError(f"default voice '{SANITY_VOICE}' not among downloaded voices")


def fetch_verbalizer():
    print("⬇️ Downloading verbalizer (m2m100 ct2 + tokenizer)...", flush=True)
    from verbalizer import Verbalizer
    Verbalizer()
    print("✅ verbalizer cached", flush=True)


def fetch_stressifier():
    print("⬇️ Downloading stanza 'uk' models via Stressifier...", flush=True)
    from ukrainian_word_stress import Stressifier
    stressify = Stressifier()
    out = stressify("Привіт, як справи?")
    print(f"✅ stressifier ready: {out!r}", flush=True)


DASH_RE = re.compile(r'[᠆‐‑‒–—―⁻₋−⸺⸻]')


def sanity_synthesis():
    """Повний прогін ланцюжка (stress → ipa → модель) на CPU прямо в білді."""
    print("🧪 Sanity synthesis on CPU...", flush=True)
    from ipa_uk import ipa
    from styletts2_inference.models import StyleTTS2
    from ukrainian_word_stress import Stressifier, StressSymbol

    stressify = Stressifier()
    model = StyleTTS2(hf_path=MODEL_REPO, device=torch.device("cpu"))

    voice_path = os.path.join(VOICES_DIR, f"{SANITY_VOICE}.pt")
    style = torch.load(voice_path, map_location="cpu").view(1, -1)

    text = "Привіт! Це тест українського синтезу мовлення."
    text = text.replace('+', StressSymbol.CombiningAcuteAccent)
    text = unicodedata.normalize("NFKC", text)
    text = DASH_RE.sub('-', text)
    text = stressify(text)
    ps = ipa(text)
    print(f"   phonemes: {ps[:120]}...", flush=True)

    tokens = model.tokenizer.encode(ps)
    wav = model(tokens, speed=1.0, s_prev=style)
    duration_sec = wav.numel() / 24000
    print(f"✅ sanity synthesis OK: {duration_sec:.2f}s of audio at 24 kHz", flush=True)
    if duration_sec < 1.0:
        raise RuntimeError(f"sanity audio too short ({duration_sec:.2f}s) — model/voice mismatch?")


fetch_tts_model()
fetch_voices()
fetch_verbalizer()
fetch_stressifier()
sanity_synthesis()

print("✅ Finished downloading and verifying all models.", flush=True)
