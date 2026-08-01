# syntax=docker/dockerfile:1

# ---- Stage 1: збірка Python-залежностей (компілятори лишаються тут) ----
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev gcc g++ ca-certificates && \
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

# python3 — для venv; ffmpeg + libsndfile1 — для декодування входу demucs
# (faster-whisper декодує через PyAV, але demucs читає mp3 через ffmpeg/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Pre-fetch Whisper model into image
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Handler code
COPY src/ /

CMD ["python", "-u", "/rp_handler.py"]
