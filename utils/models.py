import os
from pathlib import Path
from huggingface_hub import snapshot_download

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
    # patch_deepseek_v2_lite_remote_code(model_id, save_dir)


def patch_deepseek_tokenizer_config(model_id: str, save_dir: Path) -> None:
    # Post-download fix: patch DeepSeek-R1-Distill-Qwen tokenizer configuration
    if "DeepSeek-R1-Distill-Qwen" in model_id:
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
