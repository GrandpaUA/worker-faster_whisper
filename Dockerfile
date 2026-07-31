FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg wget ca-certificates && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Python deps (faster-whisper + runpod SDK)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /requirements.txt

# Pre-fetch Whisper model into image
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Handler code
COPY src/ /

CMD ["python", "-u", "/rp_handler.py"]
