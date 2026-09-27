import json
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CACHE_DIR = ROOT_DIR / ".llms"


def download_model(model_id: str) -> None:
    model_name = model_id.split("/")[-1]
    save_dir = CACHE_DIR / f"{model_name}"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Model already exists at {save_dir}")
        return

    print(f"Downloading model {model_id}...")
    snapshot_download(repo_id=model_id, local_dir=save_dir)
    print(f"Model {model_id} downloaded to {save_dir}")

    patch_deepseek_tokenizer_config(model_id, save_dir)


def patch_deepseek_tokenizer_config(model_id: str, save_dir: Path) -> None:
    if "DeepSeek-R1-Distill-Qwen" in model_id or "DeepSeek-R1-0528-Qwen3" in model_id:
        import json

        tokenizer_config_path = save_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if config_data.get("tokenizer_class") == "LlamaTokenizerFast":
                config_data["tokenizer_class"] = "Qwen2TokenizerFast"
                with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print(f"Patched tokenizer_class to Qwen2TokenizerFast for {model_id}")

    if "deepseek-llm-7b-chat" in model_id:
        import json

        tokenizer_config_path = save_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if config_data.get("tokenizer_class") == "LlamaTokenizerFast":
                config_data["tokenizer_class"] = "PreTrainedTokenizerFast"
                with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print(
                    f"Patched tokenizer_class to PreTrainedTokenizerFast for {model_id}"
                )


def get_model_generation_defaults(model_id: str) -> dict:
    """
    Fetches the generation configuration parameters for a given Hugging Face model.
    Checks generation_config.json first, then falls back to config.json.
    """
    config_data = {}

    # 1. Try to download generation_config.json first
    try:
        gen_config_path = hf_hub_download(
            repo_id=model_id, filename="generation_config.json"
        )
        with open(gen_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except EntryNotFoundError:
        # 2. If generation_config doesn't exist, fall back to core config.json
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                core_config = json.load(f)
                # Sometimes generation parameters are nested inside config.json under a key
                config_data = core_config.get("generation_config", core_config)
        except Exception as e:
            print(
                f"Warning: Could not fetch any config for {model_id}. Using strict defaults. Error: {e}"
            )
            config_data = {}
    except Exception as e:
        print(f"Warning: Error processing {model_id}: {e}")
        config_data = {}

    # 3. Extract parameters using your exact fallback rules
    parsed_config = {
        "temperature": config_data.get("temperature", 1.0),
        "top_p": config_data.get("top_p", 1.0),
        "top_k": config_data.get("top_k", 0),
        "min_p": config_data.get("min_p", 0.0),
        "repetition_penalty": config_data.get("repetition_penalty", 1.0),
        "presence_penalty": config_data.get("presence_penalty", 0.0),
        "frequency_penalty": config_data.get("frequency_penalty", 0.0),
    }

    return parsed_config


def update_benchmark_config(model_list: list, config_path: Path):
    """
    Compiles configurations for all models into the requested JSON structure.
    """

    if not config_path.exists():
        raise ValueError(f"Config path {config_path} does not exist.")

    with open(config_path, "r", encoding="utf-8") as fi:
        config_data = json.load(fi)

    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a JSON object.")

    for model in model_list:
        if model in config_data.get("models", {}):
            print(f"Model {model} already exists in config. Skipping update.")
        else:
            print(f"Fetching config for {model}...")
            config_data["models"][model] = get_model_generation_defaults(
                model.split(" ")[0]
            )

    with open(config_path, "w", encoding="utf-8") as fo:
        json.dump(config_data, fo, indent=4, ensure_ascii=False)

    print(f"Config updated and saved to {config_path}")
