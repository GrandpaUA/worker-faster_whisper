FROM runpod/base:1.1.0-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# System deps: ffmpeg for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python deps (faster-whisper + runpod SDK)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-fetch Whisper model into image
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

# Handler code
COPY src/ /

CMD ["python", "-u", "/rp_handler.py"]
