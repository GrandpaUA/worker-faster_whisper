"""RunPod serverless handler: StyleTTS2 Ukrainian TTS.

Модель: patriotyk/styletts2_ukrainian_multispeaker_istftnet (24 kHz native).
Ланцюжок — дзеркало HF space patriotyk/styletts2-ukrainian:
verbalizer (цифри) → stressify (stanza) → ipa-uk → StyleTTS2.

Контракт:
  input:  {"text": "...", "voice": "<optional, default Марина Панас>"}
  output: {"audio_base64": "<wav 22050 Hz mono 16-bit>", "duration_sec": float,
           "rtf": float, "voice": str, "gpu_name": str, ...}
  Помилка синтезу → {"error": "..."} у output (голосно, без тихих фолбеків).
"""
import base64
import io
import os
import re
import sys
import time
import traceback
import unicodedata
import wave

print("=" * 50, flush=True)
print("🚀 CONTAINER STARTING (styletts2-ua)", flush=True)
print(f"   Python: {sys.version}", flush=True)
print(f"   CWD: {os.getcwd()}", flush=True)
print("=" * 50, flush=True)

import torch
import torchaudio

MODEL_REPO = "patriotyk/styletts2_ukrainian_multispeaker_istftnet"
VOICES_DIR = "/voices"
SR_NATIVE = 24000
SR_OUT = 22050
DEFAULT_VOICE = "Марина Панас"

print("📦 Importing ipa_uk / styletts2_inference / ukrainian_word_stress...", flush=True)
from ipa_uk import ipa
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol
print("   ✅ imports OK", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"🎮 CUDA: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("⚠️ CUDA недоступна — синтез піде на CPU", flush=True)

print("🔧 Loading StyleTTS2 model...", flush=True)
_t0 = time.time()
MODEL = StyleTTS2(hf_path=MODEL_REPO, device=DEVICE)
print(f"✅ model loaded in {time.time() - _t0:.1f}s", flush=True)

print("🔧 Loading stressifier (stanza)...", flush=True)
STRESSIFY = Stressifier()
print("   ✅ stressifier OK", flush=True)

print("🔧 Loading verbalizer (m2m100 ct2)...", flush=True)
from verbalizer import Verbalizer
VERBALIZER = Verbalizer()
print("   ✅ verbalizer OK", flush=True)

print("🔧 Loading voices...", flush=True)
VOICES = {}
for _fname in sorted(os.listdir(VOICES_DIR)):
    if _fname.endswith(".pt"):
        _name = _fname[:-3]
        _style = torch.load(os.path.join(VOICES_DIR, _fname), map_location="cpu")
        VOICES[_name] = _style.view(1, -1)
print(f"✅ {len(VOICES)} voices: {sorted(VOICES)}", flush=True)
if not VOICES:
    raise RuntimeError("no voices loaded — образ зламаний (voices/ порожній)")
if DEFAULT_VOICE not in VOICES:
    raise RuntimeError(f"default voice '{DEFAULT_VOICE}' missing from {sorted(VOICES)}")

RESAMPLER = torchaudio.transforms.Resample(SR_NATIVE, SR_OUT)

import runpod
print(f"   ✅ runpod OK (version: {getattr(runpod, '__version__', 'unknown')})", flush=True)


# ---------------------------------------------------------------------------
# Текстовий конвеєр (з HF space patriotyk/styletts2-ukrainian, app.py)
# ---------------------------------------------------------------------------

DASH_RE = re.compile(r'[᠆‐‑‒–—―⁻₋−⸺⸻]')


def split_to_parts(text, group=True):
    """Розбиття на речення, як у space (короткі ≤20-символьні групи склеюються)."""
    text = re.sub(r'(\w+[^.,!:?\-])\n', r'\1. ', text)
    text = text.replace('\n', ' ')
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    last = len(text) - 1
    for i, s in enumerate(text):
        parts[index] += s
        if s in split_symbols and i < last and text[i + 1] == ' ':
            if group and len(parts[index]) <= 20:
                continue
            index += 1
            parts.append('')
    return parts


def verbalize(text):
    """Числа/акроніми → слова. Годуємо реченнями (у m2m100 ліміт входу)."""
    out = ''
    for part in split_to_parts(text, group=False):
        part = part.strip()
        if part:
            out += VERBALIZER.process_text(part)[0] + ' '
    return out.strip()


def prepare_part(part):
    """Нормалізація + наголоси, крок-в-крок як synthesize() у space."""
    t = part.strip()
    t = t.replace('"', '')
    if not t:
        return ''
    t = t.replace('+', StressSymbol.CombiningAcuteAccent)
    t = unicodedata.normalize('NFKC', t)
    t = DASH_RE.sub('-', t)
    if t[-1] not in '.?!:-':
        t += '.'
    t = re.sub(r' - ', ': ', t)
    return STRESSIFY(t)


def synthesize(text, voice_name):
    """text → waveform tensor (24 kHz, float)."""
    style = VOICES[voice_name].to(DEVICE)

    # Без verbalizer'а модель не читає цифри (README ALERTua API), тому для
    # тексту з цифрами проганяємо його; чистий текст не чіпаємо.
    if any(ch.isdigit() for ch in text):
        text = verbalize(text)

    wavs = []
    for part in split_to_parts(text):
        t = prepare_part(part)
        if not t:
            continue
        ps = ipa(t)
        if not ps.strip():
            continue
        tokens = MODEL.tokenizer.encode(ps)
        if len(tokens) == 0:
            continue
        wav = MODEL(tokens, speed=1.0, s_prev=style)
        wavs.append(wav)
    if not wavs:
        raise ValueError("synthesis produced no audio (порожня фонемна послідовність?)")
    return torch.cat(wavs).detach().cpu()


def to_wav_bytes(wav):
    """24 kHz float → WAV 22050 Hz mono 16-bit (ресемпл torchaudio)."""
    wav = RESAMPLER(wav).clamp(-1.0, 1.0)
    pcm = (wav.numpy() * 32767.0).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes(pcm.tobytes())
    return buf.getvalue(), wav.numel() / SR_OUT


def handler(job):
    job_input = job.get('input') or {}
    text = job_input.get('text')
    voice = job_input.get('voice') or DEFAULT_VOICE

    if not text or not str(text).strip():
        return {'error': 'input.text is required and must be non-empty'}
    text = str(text)
    if len(text) > 50000:
        return {'error': 'input.text too long (>50000 chars)'}
    if voice not in VOICES:
        return {'error': f"unknown voice '{voice}'. Available: {sorted(VOICES)}"}

    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'

    try:
        t0 = time.time()
        wav = synthesize(text, voice)
        audio_bytes, duration_sec = to_wav_bytes(wav)
        synth_sec = time.time() - t0
    except Exception:
        # Голосно: помилка синтезу — це output.error, не тиха деградація.
        return {'error': 'synthesis failed:\n' + traceback.format_exc()[-3000:]}

    return {
        'audio_base64': base64.b64encode(audio_bytes).decode(),
        'duration_sec': round(duration_sec, 3),
        'rtf': round(synth_sec / duration_sec, 4) if duration_sec > 0 else None,
        'synthesis_sec': round(synth_sec, 3),
        'voice': voice,
        'sample_rate': SR_OUT,
        'gpu_name': gpu_name,
    }


print("🌐 Starting runpod.serverless...", flush=True)
runpod.serverless.start({"handler": handler})
print("✅ runpod.serverless started — worker ready!", flush=True)
