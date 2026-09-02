# RunPod Worker Architecture

Цей репозиторій містить GPU-частину YouTube Voice Translator. Основний
репозиторій `youtube-translator` є клієнтом: він готує аудіо, викликає RunPod
endpoint'и, кешує результат і віддає його розширенню. Цей репозиторій є
серверною частиною RunPod: він збирає Docker images і описує handler'и, які
виконуються всередині RunPod Serverless.

## Поточні worker'и

| Worker | Image | Поточна папка | Призначення |
| --- | --- | --- | --- |
| `whisper_separate` | `drgrandpa/whisper-worker` | repo root: `Dockerfile`, `src/`, `builder/` | Faster Whisper transcription + Demucs/Roformer source separation |
| `styletts2_ua` | `drgrandpa/styletts2-ua` | `styletts2/` | Ukrainian StyleTTS2/Patriotyk TTS |

Назви `whisper_separate` і `styletts2_ua` - логічні назви продуктів. Фізично
перший worker ще лежить у корені репозиторію, другий - у `styletts2/`. Це
історичний стан, а не бажана кінцева структура.

## Межі відповідальності

`worker-faster_whisper` відповідає за:

- Docker images для RunPod.
- Handler input/output contracts.
- Prefetch моделей у image або описаний механізм runtime cache.
- GitHub Actions build pipeline.
- RunPod deploy/test tooling для endpoint'ів.

`youtube-translator` відповідає за:

- UI/extension/backend orchestration.
- `.env` endpoint IDs і runtime settings.
- Кеші транскрипції, перекладу й TTS.
- Планувальник відтворення аудіо.
- Виклики RunPod API як клієнт.

Якщо зміна стосується того, що робить Docker image або handler, вона має жити
тут. Якщо зміна стосується того, коли і як основний застосунок викликає
endpoint, вона має жити в `youtube-translator`.

## Поточний потік

### Whisper / separation

1. `youtube-translator/subgen` вирізає аудіо або chunk'и.
2. Backend відправляє job у RunPod endpoint `drgrandpa/whisper-worker`.
3. `src/rp_handler.py` валідовує input через `src/rp_schema.py`.
4. Для `task != "separate"` викликається Faster Whisper через `src/predict.py`.
5. Для `task == "separate"` викликається Demucs або Roformer.
6. Handler повертає JSON output у контрактному форматі.
7. Основний backend парсить output, кешує і продовжує pipeline.

### StyleTTS2 UA

1. `youtube-translator/tts.py` групує тексти в `texts[]`.
2. Backend відправляє job у RunPod endpoint `drgrandpa/styletts2-ua`.
3. `styletts2/src/rp_handler.py` синтезує один WAV для всього batch.
4. Handler повертає `audio_base64`, `duration_sec`, `sample_rate` і `parts`.
5. Основний backend нарізає batch WAV по `parts` і кешує сегменти.

## Поточні технічні борги

1. **Нечітка структура repo.** Один worker лежить у root, другий у `styletts2/`.
   Це працює, але не показує явно, що тут два окремі продукти.

2. **Немає контрактних тестів.** Handler contracts описані в README/коментарях,
   але не закріплені тестами без важких моделей.

3. **Whisper prefetch не синхронізований з main repo.** Основний застосунок
   використовує `RUNPOD_WHISPER_MODEL=large-v2` за замовчуванням, а поточний
   `builder/fetch_models.py` prefetch'ить `small`.

4. **Deploy tooling лежить не там.** `tools/runpod_styletts2_deploy.py` зараз у
   `youtube-translator`, хоча керує image/endpoint з цього repo.

5. **`latest` небезпечний для deploy.** Workflow пушить `latest`, а старий
   deploy script має hardcoded `drgrandpa/styletts2-ua:latest`. Після rebuild
   endpoint може продовжити працювати зі старим image cache.

6. **Root handler стартує RunPod server при import.** Це ускладнює unit tests.
   `styletts2` вже має `if __name__ == "__main__"` guard; root worker має бути
   приведений до такого ж патерну окремою поведінково нейтральною зміною.

7. **Залежності частково не pinned.** У root worker `torch`, `torchaudio`,
   `demucs`, `runpod~=1.9.0` можуть зрушити поведінку при новому build.

## Цільова структура

Фізичне переміщення файлів не є першим кроком. Його варто робити після docs,
contract tests і стабілізації build/deploy.

Бажаний кінцевий вигляд:

```text
worker-faster_whisper/
  workers/
    whisper_separate/
      Dockerfile
      builder/
      src/
      tests/
      README.md
    styletts2_ua/
      Dockerfile
      builder/
      src/
      tests/
      README.md
  docs/
    ARCHITECTURE.md
    RUNPOD_CONTRACTS.md
    DEPLOY_RUNBOOK.md
  tools/
    runpod_endpoint.py
  .github/
    workflows/
      build-workers.yml
```

Причина не рухати файли одразу: GitHub Actions, Docker contexts і deploy
scripts уже прив'язані до поточного layout. Спочатку треба зафіксувати
контракти і тести, потім переносити структуру маленькими комітами.

## Рекомендована черга змін

1. Додати docs і зафіксувати межі відповідальності.
2. Додати import-safe entrypoint guard для root worker.
3. Додати lightweight contract tests з mock'ами без завантаження моделей.
4. Синхронізувати Whisper model prefetch з production default `large-v2`.
5. Переробити build workflows: explicit worker matrix, `sha-*` tags, `latest`
   тільки як convenience.
6. Перенести RunPod deploy tooling у цей repo і вимагати explicit image tag.
7. Після цього вирішити, чи потрібен фізичний move у `workers/`.

## Джерела

- RunPod workers overview: https://docs.runpod.io/serverless/workers/overview
- RunPod handler functions: https://docs.runpod.io/serverless/workers/handler-functions
- RunPod network volumes: https://docs.runpod.io/storage/network-volumes
- Docker build best practices: https://docs.docker.com/build/building/best-practices/
- Docker Buildx Bake: https://docs.docker.com/guides/bake/
- GitHub Actions matrix: https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/running-variations-of-jobs-in-a-workflow
