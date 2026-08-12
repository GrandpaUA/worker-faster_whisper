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

MODEL_REPO = "patriotyk/styletts2_ukrainian_multispeaker_hifigan"
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


def sanity_array_mode():
    """Саніті array-режиму через реальний handler (CPU, 4 короткі тексти).

    Імпорт rp_handler безпечний: runpod-сервер там під __main__-ґвардом.
    Ловить на білді поломку chunk-шляху (chaining/trim/межі parts) і обидва
    відомі дрейфи ланцюга (інциденти живих прогонів 12.08.2026):
    - амплітудний: RMS останньої частини ≥50% RMS першої (затихання);
    - темповий: chars/sec кожної частини в [0.7×, 1.45×] медіани частин
      (розгін/уповільнення від переносу просодії).
    """
    print("🧪 Sanity array-mode via rp_handler (CPU)...", flush=True)
    import rp_handler

    texts = [
        "Привіт! Це перший тестовий сегмент.",
        "А це другий сегмент для перевірки меж.",
        "Третій сегмент перевіряє дрейф гучності.",
        "І останній сегмент для повноти картини.",
    ]
    out = rp_handler.handler({'input': {'texts': texts}})
    if 'error' in out:
        raise RuntimeError(f"array-mode sanity failed: {out['error']}")
    parts = out.get('parts') or []
    if len(parts) != 4:
        raise RuntimeError(f"array-mode sanity: expected 4 parts, got {parts!r}")
    if parts[0]['start_s'] != 0.0:
        raise RuntimeError(f"array-mode sanity: part 0 must start at 0: {parts!r}")
    for a, b in zip(parts, parts[1:]):
        if a['end_s'] != b['start_s']:
            raise RuntimeError(f"array-mode sanity: parts not contiguous: {parts!r}")
    if parts[-1]['end_s'] != out['duration_sec']:
        raise RuntimeError(f"array-mode sanity: last end {parts[-1]['end_s']}s "
                           f"!= duration_sec {out['duration_sec']}s")
    if out['duration_sec'] < 2.0:
        raise RuntimeError(f"array-mode sanity: audio too short ({out['duration_sec']}s)")

    # RMS кожної частини: ловимо дрейф амплітуди (затихання ланцюга) на білді
    import base64
    import io
    import wave
    import numpy as np
    with wave.open(io.BytesIO(base64.b64decode(out['audio_base64'])), 'rb') as w:
        if w.getframerate() != 22050 or w.getnchannels() != 1:
            raise RuntimeError(f"array-mode sanity: unexpected wav format "
                               f"{w.getframerate()}Hz/{w.getnchannels()}ch")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
    audio = audio.astype(np.float32) / 32768.0
    rms = []
    for p in parts:
        seg = audio[int(p['start_s'] * 22050):int(p['end_s'] * 22050)]
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
    print(f"   📈 RMS per part: {[round(r, 4) for r in rms]}", flush=True)
    if min(rms) <= 0.0:
        raise RuntimeError(f"array-mode sanity: silent part detected, rms={rms}")
    if rms[-1] < 0.5 * rms[0]:
        raise RuntimeError(
            f"array-mode sanity: amplitude drift — last RMS {rms[-1]:.4f} < "
            f"50% of first {rms[0]:.4f} (ланцюг затихає, s_prev-якір зламаний?)")

    # Темп кожної частини (chars/sec): ловимо розгін/уповільнення від
    # переносу просодії між items
    rates = []
    for t, p in zip(texts, parts):
        dur = p['end_s'] - p['start_s']
        if dur <= 0:
            raise RuntimeError(f"array-mode sanity: non-positive part duration: {p!r}")
        rates.append(len(t) / dur)
    med = float(np.median(rates))
    print(f"   ⏱ chars/sec per part: {[round(r, 2) for r in rates]} "
          f"(median {med:.2f})", flush=True)
    for r in rates:
        if not (0.7 * med <= r <= 1.45 * med):
            raise RuntimeError(
                f"array-mode sanity: tempo drift — {r:.2f} chars/s outside "
                f"[0.7×, 1.45×] of median {med:.2f} (просодія розганяє темп?)")

    # Порожній item — явний error, не тиха деградація
    out_err = rp_handler.handler({'input': {'texts': ['Привіт', '   ']}})
    if 'error' not in out_err:
        raise RuntimeError("array-mode sanity: empty item must return error")

    print(f"✅ array-mode sanity OK: {out['duration_sec']}s, 4 items, parts={parts}", flush=True)


fetch_tts_model()
fetch_voices()
fetch_verbalizer()
fetch_stressifier()
sanity_synthesis()
sanity_array_mode()

print("✅ Finished downloading and verifying all models.", flush=True)
