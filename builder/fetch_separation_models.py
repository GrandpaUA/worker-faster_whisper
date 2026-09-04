DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
DEFAULT_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_MODEL_DIR = "/tmp/audio-separator-models/"


def download_demucs_weights(selected_model):
    print(f"Downloading demucs {selected_model}...")
    from demucs.pretrained import get_model

    get_model(selected_model)
    print(f"Finished downloading demucs {selected_model}.")


def download_roformer_weights(selected_model):
    print(f"Downloading audio-separator model {selected_model}...")
    from audio_separator.separator import Separator

    Separator(model_file_dir=ROFORMER_MODEL_DIR).load_model(model_filename=selected_model)
    print(f"Finished downloading audio-separator model {selected_model}.")


def main():
    download_demucs_weights(DEFAULT_DEMUCS_MODEL)
    download_roformer_weights(DEFAULT_ROFORMER_MODEL)
    print("Finished downloading all separation models.")


if __name__ == "__main__":
    main()
