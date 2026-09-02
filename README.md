![Faster Whisper Logo](https://5ccaof7hvfzuzf4p.public.blob.vercel-storage.com/banner-pjbGKw0buxbWGhMVC165Gf9qgqWo7I.jpeg)

## YouTube Voice Translator workers

Цей репозиторій зараз збирає два RunPod Serverless worker'и:

- `drgrandpa/whisper-worker` - Faster Whisper transcription + Demucs/Roformer source separation.
- `drgrandpa/styletts2-ua` - Ukrainian StyleTTS2/Patriotyk TTS.

Актуальна документація проєкту:

- [Архітектура](docs/ARCHITECTURE.md)
- [RunPod contracts](docs/RUNPOD_CONTRACTS.md)
- [Deploy runbook](docs/DEPLOY_RUNBOOK.md)

Generic Faster Whisper documentation нижче - історична upstream-документація
worker'а. Поточним контрактом продукту вважати файли в `docs/`.

[Faster Whisper](https://github.com/guillaumekln/faster-whisper) is designed to process audio files using various Whisper models, with options for transcription formatting, language translation and more.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-faster_whisper)](https://www.runpod.io/console/hub/runpod-workers/worker-faster_whisper)

---

## Models

- tiny
- base
- small
- medium
- large-v1
- large-v2
- large-v3
- distil-large-v2
- distil-large-v3
- turbo

## Input

| Input                               | Type  | Description                                                                                                                                                            |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio`                             | Path  | URL to Audio file                                                                                                                                                      |
| `audio_base64`                      | str   | Base64-encoded audio file                                                                                                                                              |
| `model`                             | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "distil-large-v2", "distil-large-v3", "turbo". Default: "base" |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                  |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                                                                                         |
| `translation`                       | str   | Choose the format for the translation. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                    |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                                                                                |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                                                                            |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                               |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                                                                                   |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: None                                                                                                         |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: None                                                                                                       |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                                                                           |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                                                                               |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True                                                                       |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                                                                                        |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4                                                                    |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0                                                                   |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6                                                               |
| `enable_vad`                        | bool  | If True, use the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model. Default: False               |
| `word_timestamps`                   | bool  | If True, include word timestamps in the output. Default: False                                                                                                         |

### Example

The following inputs can be used for testing the model:

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "turbo"
  }
}
```

producing an output like this:

```json
{
  "segments": [
    {
      "id": 1,
      "seek": 106,
      "start": 0.11,
      "end": 3.11,
      "text": " Hello and welcome!",
      "tokens": [50364, 25, 7, 287, 50514],
      "temperature": 0.1,
      "avg_logprob": -0.8348079785480325,
      "compression_ratio": 0.5789473684210527,
      "no_speech_prob": 0.1453857421875
    }
  ],
  "detected_language": "en",
  "transcription": "Hello and welcome!",
  "translation": null,
  "device": "cuda",
  "model": "turbo",
  "translation_time": 0.3796223163604736
}
```

## Source separation (`task: "separate"`)

GPU separation of a song into vocals and instrumental. Besides Demucs
(`engine: "demucs"`, default, `htdemucs_ft`), the worker supports
`engine: "roformer"` — [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
running a BS-Roformer model. Default roformer model is
`model_bs_roformer_ep_317_sdr_12.9755.ckpt` (top vocals+instrumental SDR,
pre-fetched into the image); any other model filename from the
audio-separator registry can be passed via `roformer_model`.

| Input            | Type     | Description                                                              |
| ---------------- | -------- | ------------------------------------------------------------------------ |
| `task`           | str      | `"separate"` instead of transcription. Default: `"transcribe"`            |
| `audio`          | Path     | URL to audio file (exactly one of `audio` / `audio_base64`)               |
| `audio_base64`   | str      | Base64-encoded audio file                                                 |
| `engine`         | str      | `"demucs"` (default) or `"roformer"`                                      |
| `demucs_model`   | str      | Demucs model name. Default: `"htdemucs_ft"`                               |
| `roformer_model` | str      | audio-separator model file. Default: `"model_bs_roformer_ep_317_sdr_12.9755.ckpt"` |
| `return_stems`   | bool     | `false` returns metadata only — full base64 stems can exceed RunPod's ~20MB response limit. Default: `true` |

Both engines return the same contract: `{separation_sec, model, cuda_available,
gpu_name, vocals_base64/no_vocals_base64}` (or `*_bytes` sizes when
`return_stems` is false). The roformer branch additionally reports
`engine: "roformer"`, `torch_device` and `onnx_execution_provider` so GPU
usage can be verified, and returns `{error, stderr}` on failure.

```json
{
  "input": {
    "task": "separate",
    "engine": "roformer",
    "audio_base64": "<base64 wav/mp3>"
  }
}
```
