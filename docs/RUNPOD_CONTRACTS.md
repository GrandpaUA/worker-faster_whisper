# RunPod Contracts

Цей документ описує JSON-контракти між `youtube-translator` і RunPod
worker'ами. Він має бути первинним місцем для перевірки сумісності перед
змінами у handler'ах або клієнтському коді.

## Загальні правила

- Усі endpoint'и приймають RunPod job envelope:

```json
{
  "id": "runpod-job-id",
  "input": {}
}
```

- Handler повертає plain JSON object.
- Помилка worker'а повертається як `{"error": "..."}`.
- `output: null` від RunPod не є валідним успіхом для нашого продукту. Клієнт
  має трактувати це як гучну помилку.
- Для довгих задач клієнт використовує async `/run` + `/status/{job_id}`.

## `whisper_separate`: transcription

### Input

```json
{
  "input": {
    "audio": "https://example.com/audio.wav",
    "model": "large-v2",
    "transcription": "plain_text",
    "translate": false,
    "translation": "plain_text",
    "language": null,
    "temperature": 0,
    "best_of": 5,
    "beam_size": 5,
    "patience": 1,
    "length_penalty": 0,
    "suppress_tokens": "-1",
    "initial_prompt": null,
    "condition_on_previous_text": true,
    "temperature_increment_on_fallback": 0.2,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "enable_vad": false,
    "word_timestamps": false
  }
}
```

`audio_base64` може використовуватись замість `audio`. Рівно одне з полів
`audio` або `audio_base64` має бути задане.
Якщо `model` не заданий, worker використовує `large-v2`.

### Output

```json
{
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.2,
      "text": " Hello.",
      "tokens": [50364],
      "temperature": 0.0,
      "avg_logprob": -0.2,
      "compression_ratio": 1.1,
      "no_speech_prob": 0.01
    }
  ],
  "detected_language": "en",
  "transcription": "Hello.",
  "translation": null,
  "device": "cuda",
  "model": "large-v2"
}
```

Якщо `word_timestamps=true`, output також містить:

```json
{
  "word_timestamps": [
    {"word": "Hello", "start": 0.1, "end": 0.6}
  ]
}
```

### Production expectation

Основний repo зараз очікує production default `large-v2`. Звичайний image
prefetch'ить `large-v2`; інші моделі для експериментів треба задавати явно.
Тиха runtime дозагрузка непередбаченої моделі небажана, бо маскує cold-start
проблему.

## `whisper_separate`: source separation

### Input

```json
{
  "input": {
    "task": "separate",
    "audio_base64": "<base64 wav/mp3>",
    "engine": "roformer",
    "return_stems": true
  }
}
```

Поля:

| Field | Default | Meaning |
| --- | --- | --- |
| `task` | `"transcribe"` | Для separation має бути `"separate"` |
| `audio` / `audio_base64` | none | Рівно одне джерело аудіо |
| `engine` | `"demucs"` | `"demucs"` або `"roformer"` |
| `demucs_model` | `"htdemucs_ft"` | Demucs model |
| `roformer_model` | `"model_bs_roformer_ep_317_sdr_12.9755.ckpt"` | audio-separator model |
| `return_stems` | `true` | `false` повертає тільки metadata і розміри |

### Output

```json
{
  "separation_sec": 12.34,
  "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
  "engine": "roformer",
  "cuda_available": true,
  "gpu_name": "NVIDIA RTX 4000 Ada Generation",
  "torch_device": "cuda",
  "onnx_execution_provider": "CUDAExecutionProvider",
  "vocals_base64": "<base64 mp3>",
  "no_vocals_base64": "<base64 mp3>"
}
```

Якщо `return_stems=false`, очікуються `vocals_bytes` і `no_vocals_bytes`
замість base64 stem'ів.

## `styletts2_ua`: single text

### Input

```json
{
  "input": {
    "text": "Привіт.",
    "voice": "Марина Панас"
  }
}
```

`voice` optional. Якщо не заданий, worker використовує default voice.

### Output

```json
{
  "audio_base64": "<base64 wav>",
  "duration_sec": 1.23,
  "rtf": 0.2,
  "synthesis_sec": 0.25,
  "voice": "Марина Панас",
  "sample_rate": 22050,
  "gpu_name": "NVIDIA RTX 4000 Ada Generation"
}
```

WAV contract: mono, 16-bit PCM, 22050 Hz.

## `styletts2_ua`: batch texts

### Input

```json
{
  "input": {
    "texts": [
      "Перше речення.",
      "Друге речення."
    ],
    "voice": "Марина Панас"
  }
}
```

Якщо задані і `texts`, і `text`, перемагає `texts`.

### Output

```json
{
  "audio_base64": "<base64 wav>",
  "duration_sec": 3.45,
  "rtf": 0.2,
  "synthesis_sec": 0.7,
  "voice": "Марина Панас",
  "sample_rate": 22050,
  "gpu_name": "NVIDIA RTX 4000 Ada Generation",
  "parts": [
    {"index": 0, "start_s": 0.0, "end_s": 1.4},
    {"index": 1, "start_s": 1.4, "end_s": 3.45}
  ]
}
```

`parts` - межі кожного input item у фінальному WAV. Основний backend має
нарізати audio за цими межами і кешувати сегменти окремо.

## Що має покривати contract test suite

- Handler import не стартує RunPod server.
- Transcription без `model` використовує default `large-v2`.
- Input validation відкидає одночасні `audio` і `audio_base64`.
- Transcription output має `segments`, `detected_language`, `model`, `device`.
- Separation output для `return_stems=false` не містить base64 stem'ів.
- StyleTTS2 batch output має моно WAV 22050 Hz і `parts` з монотонними межами.
- Worker error повертається як `{"error": ...}`, а не тиха деградація.
