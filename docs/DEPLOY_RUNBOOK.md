# RunPod Deploy Runbook

Ціль deploy process: endpoint має явно показувати, який саме image запущений.
`latest` може існувати для ручного тесту, але не має бути production deploy
target.

## Поточні images

| Worker | Docker image | GitHub workflow |
| --- | --- | --- |
| `whisper_separate` | `drgrandpa/whisper-worker` | `.github/workflows/build-image.yml` |
| `styletts2_ua` | `drgrandpa/styletts2-ua` | `.github/workflows/build-styletts2.yml` |

## Правило image tags

Обов'язковий tag для deploy:

```text
drgrandpa/<image>:sha-<short_commit_sha>
```

Допустимий convenience tag:

```text
drgrandpa/<image>:latest
```

`latest` не використовувати для оновлення endpoint. Причина проста: mutable tag
не гарантує, що RunPod worker підтягне саме новий image. У нас уже був інцидент,
коли rebuild відбувся, але endpoint продовжив відповідати старим handler'ом.

## Безпечний ручний deploy

1. Переконатись, що потрібний commit запушений у `worker-faster_whisper`.
2. Дочекатись успішного GitHub Actions build.
3. Взяти `sha-*` image tag з build summary або Docker Hub.
4. Оновити RunPod template/endpoint саме на цей image tag.
5. Перевірити endpoint через реальний RunPod `/run` + `/status`, не тільки
   через UI status.
6. Перевірити, що output містить продукт: transcription segments, stems або WAV.

## Whisper model prefetch

Звичайний build `drgrandpa/whisper-worker` prefetch'ить `large-v2`, бо це
production default основного застосунку. Для експериментального image можна
передати Docker build arg:

```text
WHISPER_MODELS=small,large-v2
```

Не додавати кілька Whisper моделей без причини: кожна модель збільшує image на
гігабайти і подовжує build/pull.

## Що вважати успішним deploy

Для `whisper_separate`:

- endpoint приймає `audio` або `audio_base64`;
- `model` у output відповідає запитаній моделі;
- `device` показує `cuda`, якщо endpoint має GPU;
- `segments` непорожній для тестового аудіо з мовою;
- для separation `return_stems=false` повертаються розміри stem'ів.

Для `styletts2_ua`:

- endpoint повертає `audio_base64`;
- WAV mono, 16-bit PCM, 22050 Hz;
- `duration_sec > 0`;
- batch mode повертає `parts`;
- `parts` монотонні і не виходять за межі WAV duration.

## Secrets

Потрібні GitHub repository secrets для build workflows:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

`RUNPOD_API_KEY` не потрібен для build. Його додавати в GitHub secrets тільки
якщо ми свідомо робимо CI-driven deploy. Для поточного безпечного етапу deploy
має бути ручним або через локальний tool, який читає `.env`.

## Endpoint settings

Для першої стабілізації не міняти одночасно image і endpoint resources. Якщо
міняємо image - не міняємо GPU pool, disk, workers, timeout. Якщо міняємо
resources - не міняємо image. Так легше зрозуміти причину регресу.

Network volumes:

- можуть зменшити download/cache час;
- прив'язують worker до датацентру;
- можуть зменшити доступність GPU;
- не синхронізуються автоматично між датацентрами.

Тому network volume - окремий експеримент після image-level prefetch і вимірів
cold start.

FlashBoot:

- може зменшити cold start;
- не замінює model prefetch;
- не доводить, що endpoint працює правильним image.

Тому FlashBoot - endpoint optimization, а не заміна відтворюваного deploy.

## Рекомендована наступна перебудова workflow

Поточні два workflow можна залишити тимчасово, але цільовий стан:

- один `build-workers.yml`;
- matrix по worker'ах;
- explicit context/file/image per worker;
- `sha-*` tag обов'язковий;
- `latest` optional;
- build cache зі scope per worker;
- top-level `permissions: contents: read`;
- manual `workflow_dispatch` з вибором worker і `nocache`.

Приклад логіки matrix:

```yaml
strategy:
  matrix:
    worker:
      - name: whisper_separate
        context: .
        dockerfile: Dockerfile
        image: drgrandpa/whisper-worker
        cache_scope: whisper
      - name: styletts2_ua
        context: styletts2
        dockerfile: styletts2/Dockerfile
        image: drgrandpa/styletts2-ua
        cache_scope: styletts2
```

## Rollback

Rollback - це не rebuild. Rollback має оновити endpoint на попередній відомий
добрий `sha-*` image tag.

Перед rollback треба зберегти:

- endpoint id;
- bad image tag;
- previous good image tag;
- RunPod job id, який показав регрес;
- короткий симптом.

## Джерела

- RunPod deploy workers: https://docs.runpod.io/serverless/workers/deploy
- RunPod manage endpoints: https://docs.runpod.io/sdks/graphql/manage-endpoints
- RunPod send requests: https://docs.runpod.io/serverless/endpoints/send-requests
- Docker metadata action: https://github.com/docker/metadata-action
- Docker cache with GitHub Actions: https://docs.docker.com/build/ci/github-actions/cache/
- GitHub Actions secure use: https://docs.github.com/en/actions/reference/security/secure-use
