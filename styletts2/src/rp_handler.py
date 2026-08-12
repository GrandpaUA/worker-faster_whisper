"""RunPod serverless handler: StyleTTS2 Ukrainian TTS.

Модель: patriotyk/styletts2_ukrainian_multispeaker_hifigan (24 kHz native).
ВАЖЛИВО: саме hifigan — HF space patriotyk/styletts2-ukrainian ганяє її
('.../multispeaker' → redirect на _hifigan), і всі 31 голосів зі space
зняті під неї. istftnet-варіант з цими голосами = деградація тембру
(аудит 11.08.2026).
Ланцюжок — дзеркало HF space patriotyk/styletts2-ukrainian:
verbalizer (цифри) → stressify (stanza) → ipa-uk → StyleTTS2.

Контракт:
  input (single): {"text": "...", "voice": "<optional, default Марина Панас>"}
  output: {"audio_base64": "<wav 22050 Hz mono 16-bit>", "duration_sec": float,
           "rtf": float, "voice": str, "gpu_name": str, ...}
  input (chunk):  {"texts": ["...", ...], "voice": "<optional>"} —
    items склеюються в один WAV; між items prosody chaining (s_prev = стиль,
    знятий extract_voice_features з аудіо попереднього item'а); тиша по краях
    кожного item'а ріжеться до ≤30мс lead / ≤120мс trail (поріг −50dBFS) +
    10мс лінійні фейди на краях. В output додається
    "parts": [{"index": i, "start_s": ..., "end_s": ...}] — межі КОЖНОГО item
    у фінальній шкалі 22050 Гц (item із кількох речень = від старту першого
    до кінця останнього). Якщо задані і "texts", і "text" — "texts" виграє.
  Помилка синтезу → {"error": "..."} у output (голосно, без тихих фолбеків).
"""
import base64
import io
import os
import re
import soundfile as sf
import sys
import tempfile
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

MODEL_REPO = "patriotyk/styletts2_ukrainian_multispeaker_hifigan"
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

    # Без verbalizer'а модель не читає цифри й акроніми (README space).
    # Тригер: цифри АБО латиниця (NASA, GPU, C++) — сира латиниця в ipa_uk
    # дає сміття (аудит 11.08.2026: раніше було лише isdigit → акроніми
    # без цифр пролітали повз вербалізатор).
    if re.search(r'[0-9A-Za-z]', text):
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


# ---------------------------------------------------------------------------
# Array-режим (chunk): {"texts": [...]} → один WAV + межі item'ів
# ---------------------------------------------------------------------------

TRIM_LEAD_MS = 30     # скільки тиші лишати на початку item'а
TRIM_TRAIL_MS = 120   # скільки тиші лишати в кінці item'а
TRIM_DBFS = -50.0     # поріг "тиші" (dBFS)
FADE_MS = 10          # лінійний фейд на краях після trim


def trim_silence(wav, sr):
    """Зрізати lead-тишу до ≤30мс і trail-тишу до ≤120мс (поріг −50dBFS)."""
    threshold = 10 ** (TRIM_DBFS / 20.0)
    active = (wav.abs() > threshold).nonzero().squeeze(1)
    if active.numel() == 0:
        # item повністю тихий — стягуємо в lead+trail залишок
        return wav[:(TRIM_LEAD_MS + TRIM_TRAIL_MS) * sr // 1000]
    first = int(active[0])
    last = int(active[-1])
    start = max(0, first - TRIM_LEAD_MS * sr // 1000)
    end = min(wav.numel(), last + 1 + TRIM_TRAIL_MS * sr // 1000)
    return wav[start:end]


def apply_fades(wav, sr):
    """10мс лінійний fade-in/out на краях частини (після trim)."""
    n_fade = FADE_MS * sr // 1000
    if wav.numel() < 2 * n_fade:
        n_fade = wav.numel() // 2
    if n_fade > 0:
        ramp = torch.linspace(0.0, 1.0, n_fade, dtype=wav.dtype)
        wav = wav.clone()
        wav[:n_fade] *= ramp
        wav[-n_fade:] *= ramp.flip(0)
    return wav


def style_from_item_wav(wav24):
    """s_prev для наступного item'а: extract_voice_features з аудіо поточного.

    Через тимчасовий WAV: librosa.load (0.11+) не приймає ndarray, а шлях
    читає разом із заголовком семплрейту — це ж файл-шлях, що й у HF space.
    """
    fd, path = tempfile.mkstemp(suffix='.wav', prefix='st2_chain_')
    os.close(fd)
    try:
        sf.write(path, wav24.clamp(-1.0, 1.0).numpy(), SR_NATIVE)
        return MODEL.extract_voice_features(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def synthesize_item(text, s_prev):
    """Один item → waveform 24 kHz. Пайплайн ідентичний synthesize(), але
    s_prev передається звоні (prosody chaining між items)."""
    if re.search(r'[0-9A-Za-z]', text):
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
        wav = MODEL(tokens, speed=1.0, s_prev=s_prev)
        wavs.append(wav)
    if not wavs:
        raise ValueError("synthesis produced no audio (порожня фонемна послідовність?)")
    return torch.cat(wavs).detach().cpu()


def synthesize_chain(texts, voice_name):
    """Array-режим: (wav 22050 Hz, parts). parts[i] — межі item'а i.

    Перший item — голосовий .pt вектор; для item'а k>1 s_prev = стиль, знятий
    зі згенерованого аудіо item'а k-1. Перед склейкою кожного item'а —
    trim тиші + фейди.
    """
    s_prev = VOICES[voice_name].to(DEVICE)
    out = []
    parts = []
    cum = 0
    last = len(texts) - 1
    for i, text in enumerate(texts):
        try:
            wav24 = synthesize_item(text, s_prev)
        except Exception as exc:
            raise RuntimeError(f"item {i} ({text[:60]!r}): {exc}") from exc
        if i < last:
            s_prev = style_from_item_wav(wav24)
        wav24 = trim_silence(wav24, SR_NATIVE)
        wav24 = apply_fades(wav24, SR_NATIVE)
        wav22 = RESAMPLER(wav24).clamp(-1.0, 1.0)
        n = wav22.numel()
        parts.append({
            'index': i,
            'start_s': round(cum / SR_OUT, 3),
            'end_s': round((cum + n) / SR_OUT, 3),
        })
        out.append(wav22)
        cum += n
    return torch.cat(out), parts


def handle_texts(job_input, texts):
    """Array-режим: {"texts": [...], "voice": ...} → один WAV + parts."""
    voice = job_input.get('voice') or DEFAULT_VOICE

    if not isinstance(texts, (list, tuple)):
        return {'error': 'input.texts must be an array of strings'}
    if len(texts) == 0:
        return {'error': 'input.texts must contain at least one item'}
    for i, item in enumerate(texts):
        if not isinstance(item, str) or not item.strip():
            return {'error': f'input.texts[{i}] must be a non-empty string'}
    if sum(len(t) for t in texts) > 50000:
        return {'error': 'input.texts too long (>50000 chars total)'}
    if voice not in VOICES:
        return {'error': f"unknown voice '{voice}'. Available: {sorted(VOICES)}"}

    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'

    try:
        t0 = time.time()
        wav, parts = synthesize_chain(list(texts), voice)
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
        'parts': parts,
    }


def handler(job):
    job_input = job.get('input') or {}
    texts = job_input.get('texts')
    if texts is not None:
        return handle_texts(job_input, texts)
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


# __main__-ґвард: запуск сервера лише коли файл — точка входу (CMD у Docker).
# Під білдом fetch_models.py імпортує цей модуль для саніті array-режиму —
# без ґварда імпорт підняв би runpod-сервер посеред docker build.
if __name__ == "__main__":
    print("🌐 Starting runpod.serverless...", flush=True)
    runpod.serverless.start({"handler": handler})
    print("✅ runpod.serverless started — worker ready!", flush=True)
