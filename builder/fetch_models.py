import os

# Only download the model(s) actually used by the backend.
# Default matches RUNPOD_WHISPER_MODEL=large-v2 in youtube-translator/subgen/settings.py.
# Add more only if needed — each model adds GBs to the image.
DEFAULT_WHISPER_MODELS = "large-v2"
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
DEFAULT_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_MODEL_DIR = "/tmp/audio-separator-models/"


def parse_whisper_models(raw_models: str | None) -> list[str]:
    raw_models = raw_models if raw_models is not None else DEFAULT_WHISPER_MODELS
    model_names = [name.strip() for name in raw_models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("WHISPER_MODELS must contain at least one model name")
    return model_names


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    from faster_whisper.utils import download_model
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


def download_demucs_weights(selected_model):
    """
    Download Demucs model weights (cached in the image so cold start is fast).
    """
    print(f"Downloading demucs {selected_model}...")
    from demucs.pretrained import get_model
    get_model(selected_model)
    print(f"Finished downloading demucs {selected_model}.")


def download_roformer_weights(selected_model):
    """
    Download audio-separator (BS-Roformer) weights into the image so cold
    start doesn't pull ~600MB. model_file_dir must match ROFORMER_MODEL_DIR
    in rp_handler.py (/tmp/audio-separator-models/).
    """
    print(f"Downloading audio-separator model {selected_model}...")
    from audio_separator.separator import Separator
    Separator(model_file_dir=ROFORMER_MODEL_DIR).load_model(model_filename=selected_model)
    print(f"Finished downloading audio-separator model {selected_model}.")


def main():
    for model_name in parse_whisper_models(os.getenv("WHISPER_MODELS")):
        download_model_weights(model_name)

    # Pre-fetch Demucs htdemucs_ft weights used by the separation task.
    download_demucs_weights(DEFAULT_DEMUCS_MODEL)

    # Pre-fetch the default engine='roformer' model (vocals + instrumental).
    download_roformer_weights(DEFAULT_ROFORMER_MODEL)

    print("Finished downloading all models.")


if __name__ == "__main__":
    main()
