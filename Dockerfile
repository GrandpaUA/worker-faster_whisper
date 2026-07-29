FROM runpod/base:1.1.0-cuda1281-ubuntu2204
# System deps for audio processing
RUN apt-get update && \
          apt-get install -y --no-install-recommends ffmpeg libgl1 libx11-6 && \
          rm -rf /var/lib/apt/lists/*
# Python dependencies
       COPY builder/requirements.txt /requirements.txt
      RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system
      
      # Pre-fetch models for faster cold starts
      COPY builder/fetch_models.py /fetch_models.py
      RUN python /fetch_models.py && rm /fetch_models.py

      # Copy handler code
      COPY src/ /
      
      CMD python -u /rp_handler.py

# Set default command
CMD python -u /rp_handler.py
