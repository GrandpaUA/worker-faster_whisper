import os

# Only download the Whisper model(s) actually used by the backend.
# Default matches RUNPOD_WHISPER_MODEL=large-v2 in youtube-translator/subgen/settings.py.
DEFAULT_WHISPER_MODELS = "large-v2"


def parse_whisper_models(raw_models: str | None) -> list[str]:
    raw_models = raw_models if raw_models is not None else DEFAULT_WHISPER_MODELS
    model_names = [name.strip() for name in raw_models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("WHISPER_MODELS must contain at least one model name")
    return model_names


def download_model_weights(selected_model):
    print(f"Downloading {selected_model}...")
    from faster_whisper.utils import download_model

    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


def main():
    for model_name in parse_whisper_models(os.getenv("WHISPER_MODELS")):
        download_model_weights(model_name)

    print("Finished downloading all Whisper models.")


if __name__ == "__main__":
    main()
