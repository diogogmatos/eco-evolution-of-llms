import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

torch.cuda.empty_cache()

from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForMultimodalLM,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from huggingface_hub import HfApi, snapshot_download
import multiprocessing as mp

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CACHE_DIR = ROOT_DIR / ".llms"


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


def patch_deepseek_v2_lite_remote_code(model_id: str, save_dir: Path) -> None:
    # Post-download fix: patch DeepSeek-V2-Lite-Chat remote code for compatibility with current Hugging Face Transformers version.
    if "DeepSeek-V2-Lite-Chat" in model_id:
        with open(
            ROOT_DIR / "patches/DeepSeek-V2-Lite-Chat/modeling_deepseek.py",
            "r",
            encoding="utf-8",
        ) as f:
            patch_code = f.read()

        modeling_file_path = save_dir / "modeling_deepseek.py"

        with open(modeling_file_path, "w", encoding="utf-8") as f:
            f.write(patch_code)

        print(
            f"Applied remote code patch for DeepSeek-V2-Lite-Chat at {modeling_file_path}"
        )


def get_model(model_id: str) -> None:
    # Try to quantize the model locally first,
    # which will also download the full-precision model if it's not already cached.
    model_name = model_id.split("/")[-1]

    save_dir = CACHE_DIR / f"{model_name}"
    if save_dir.exists() and save_dir.is_dir():
        print(f"Model already exists at {save_dir}")
    else:
        download_model(model_id=model_id)

    save_dir = CACHE_DIR / f"{model_name}-fp8"
    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
    else:
        try:
            # Run quantization in a separate process to isolate GPU memory usage and ensure it gets released after completion.
            p = mp.Process(target=quantize_model, args=(model_id,))
            p.start()
            p.join()
        except:
            raise Exception(
                f"Failed to quantize {model_id} locally. Please ensure the model can be loaded and quantized with the current setup. Consider downloading a pre-quantized FP8-DYNAMIC version from Hugging Face if available."
            )


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
    patch_deepseek_v2_lite_remote_code(model_id, save_dir)


def download_quantized_model(model_id: str) -> None:
    api = HfApi()
    model_provider = model_id.split("/")[0]
    model_name = model_id.split("/")[-1]
    save_dir = CACHE_DIR / f"{model_name}-fp8"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
        return

    providers = [model_provider, "RedHatAI", "arcee-ai", "neuralmagic", "unsloth"]

    target_repo_id = model_id

    # Search for available quantized versions linked to this base model
    model = api.model_info(
        model_id
    )  # The model itself sometimes is already quantized (ex: mistralai/Ministral-3-14B-Instruct-2512)
    models_linked = list(api.list_models(filter=f"base_model:quantized:{model_id}"))

    # Combine and deduplicate
    all_candidates = [m for m in ([model] + models_linked)]

    found = False

    # 1. Search for FP8-Dynamic models from preferred providers
    for provider in providers:
        for m in all_candidates:
            if model_name.lower() in m.id.lower().split("/")[-1]:
                if (
                    m.id.lower().startswith(f"{provider.lower()}/")
                    and "-fp8" in m.id.lower()
                    and "-dynamic" in m.id.lower()
                ):
                    target_repo_id = m.id
                    found = True
                    break
                elif m.id.lower().startswith(f"{provider.lower()}/") and (
                    (m.tags is not None and "fp8" in m.tags) or ("-fp8" in m.id.lower())
                ):
                    target_repo_id = m.id
                    found = True
                    break
        if found:
            break

    # 2. If no FP8 models found from preferred providers, search for any FP8 model
    if not found:
        for m in all_candidates:
            if model_name.lower() in m.id.lower().split("/")[-1]:
                if "-fp8" in m.id.lower() and "-dynamic" in m.id.lower():
                    target_repo_id = m.id
                    found = True
                    break
                elif (m.tags is not None and "fp8" in m.tags) or (
                    "-fp8" in m.id.lower()
                ):
                    target_repo_id = m.id
                    found = True
                    break

    if found:
        print(f"Found quantized model: {target_repo_id}")
        snapshot_download(repo_id=target_repo_id, local_dir=save_dir)
        print(f"Model {target_repo_id} downloaded to {save_dir}")

        patch_deepseek_tokenizer_config(model_id, save_dir)
    else:
        print(f"No FP8 version found for {model_id}.")
        raise Exception(f"No FP8 version found for {model_id}.")


def quantize_model(model_id: str) -> None:
    model_name = model_id.split("/")[-1]
    model_dir = CACHE_DIR / model_name
    save_dir = CACHE_DIR / f"{model_name}-fp8"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
        return

    print(f"Loading {model_name}...")

    # Choose the correct Auto class based on the model's architecture
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if "ForConditionalGeneration" in config.architectures[0]:  # type: ignore
        model_class = AutoModelForMultimodalLM
    else:
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 2. Configure the quantization algorithm and scheme.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*router$",
            "re:.*embed.*",
            "re:.*norm.*",
            # --- Vision Components ---
            "re:.*vision_tower.*",
            "re:.*vision_model.*",
            "re:visual.*",
            "re:.*visual.*",
            "re:model.visual.*",
            # --- Audio Components ---
            "re:.*audio_tower.*",
            "re:.*audio_model.*",
            "re:.*audio_encoder.*",
            "re:audio.*",
            "re:.*audio.*",
            "re:model.audio.*",
            # --- General Multimodal Projectors ---
            "re:.*multi_modal_projector.*",
            "re:.*audio_projector.*",
            "re:.*vision_projector.*",
        ],
    )

    print("Applying quantization...")

    # 3. Apply quantization.

    # Temporarily hide GPUs from PyTorch to forcefully prevent llmcompressor's
    # automated offload-mapping from silently routing tensors to cuda:0.
    orig_cuda_is_available = torch.cuda.is_available
    orig_cuda_device_count = torch.cuda.device_count
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    try:
        oneshot_result = oneshot(model=model, recipe=recipe, pipeline="datafree")  # type: ignore
    finally:
        torch.cuda.is_available = orig_cuda_is_available
        torch.cuda.device_count = orig_cuda_device_count

    # 4. Save to disk in compressed-tensors format.
    print(f"Saving quantized model to {save_dir}...")

    # Patch for DeepSeek remote code compatibility with transformers saving
    import transformers.modeling_utils

    original_get_tied_weight_keys = transformers.modeling_utils._get_tied_weight_keys

    def _patched_get_tied_weight_keys(module):
        tied_weight_keys = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", {}) or {}
            if isinstance(tied, list):
                tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied])
            elif hasattr(tied, "keys"):
                tied_weight_keys.extend(
                    [f"{name}.{k}" if name else k for k in tied.keys()]
                )
        return tied_weight_keys

    # Apply monkey patch
    transformers.modeling_utils._get_tied_weight_keys = _patched_get_tied_weight_keys

    try:
        model.save_pretrained(save_dir)
    finally:
        # Restore original function
        transformers.modeling_utils._get_tied_weight_keys = (
            original_get_tied_weight_keys
        )

    tokenizer.save_pretrained(save_dir)

    patch_deepseek_tokenizer_config(model_id, save_dir)

    # Cleanup to prevent GPU memory accumulation
    del model
    del tokenizer
    del oneshot_result
    del recipe
    import gc

    gc.collect()
    torch.cuda.empty_cache()
