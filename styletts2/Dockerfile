# syntax=docker/dockerfile:1

# StyleTTS2 Ukrainian (patriotyk/styletts2_ukrainian_multispeaker_istftnet)
# для сліпого A/B проти Edge-TTS. Інференс — дзеркало HF space
# patriotyk/styletts2-ukrainian (styletts2-inference + ipa-uk + ukrainian-word-stress).

# ---- Stage 1: збірка Python-залежностей (компілятори лишаються тут) ----
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# git — для pip-залежностей з GitHub (styletts2-inference, ipa-uk, ukrainian-word-stress)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev gcc g++ ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ---- Stage 2: чистий runtime ----
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# python3 — для venv; libsndfile1 — soundfile/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Handler + verbalizer копіюємо ДО префетчу: fetch_models.py імпортує verbalizer
# і ганяє sanity-синтез, тож код і моделі перевіряються ще на білді.
COPY src/ /

COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Runtime без мережних завантажень: всі моделі вже в образі (принцип: фейли голосно —
# якщо кеш неповний, hub впаде з помилкою, а не тихо підтягне щось у рантаймі).
ENV HF_HUB_OFFLINE=1

CMD ["python", "-u", "/rp_handler.py"]
