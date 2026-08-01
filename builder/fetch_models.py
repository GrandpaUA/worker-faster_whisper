from faster_whisper.utils import download_model

# Only download the model(s) actually used by the backend.
# Default: RUNPOD_WHISPER_MODEL=small (see subtitles.py)
# Add more only if needed — each model adds GBs to the image.
model_names = [
    "small",
]


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


# Loop through models sequentially
for model_name in model_names:
    download_model_weights(model_name)


def download_demucs_weights(selected_model):
    """
    Download Demucs model weights (cached in the image so cold start is fast).
    """
    print(f"Downloading demucs {selected_model}...")
    from demucs.pretrained import get_model
    get_model(selected_model)
    print(f"Finished downloading demucs {selected_model}.")


# Pre-fetch Demucs htdemucs_ft weights used by the separation task.
download_demucs_weights("htdemucs_ft")

print("Finished downloading all models.")
