FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python 3.10 + ffmpeg for audio
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Python deps (faster-whisper + runpod SDK)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-fetch Whisper models into image
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Handler code
COPY src/ /

CMD python -u /rp_handler.py
