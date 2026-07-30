FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# System deps for audio
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python deps (faster-whisper only - PyTorch + runpod SDK already in base)
COPY builder/requirements.txt /requirements.txt
RUN uv pip install --no-cache-dir --system -r /requirements.txt

# Pre-fetch Whisper models into image
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Handler code
COPY src/ /

CMD python -u /rp_handler.py
