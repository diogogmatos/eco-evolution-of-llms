import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

torch.cuda.empty_cache()

from pathlib import Path

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor import model_free_ptq
from llmcompressor.modifiers.quantization import QuantizationModifier

from huggingface_hub import HfApi, snapshot_download

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DEFAULT_CACHE = ROOT_DIR / ".llms"


def download_model(model_id: str, cache_dir: Path | str = DEFAULT_CACHE) -> None:
    model_name = model_id.split("/")[-1]
    save_dir = Path(cache_dir) / f"{model_name}"
    snapshot_download(repo_id=model_id, local_dir=save_dir)
    print(f"Model {model_id} downloaded to {save_dir}")


def download_quantized_model(
    model_id: str, cache_dir: Path | str = DEFAULT_CACHE
) -> None:
    api = HfApi()
    model_provider = model_id.split("/")[0]
    model_name = model_id.split("/")[-1]
    save_dir = Path(cache_dir) / f"{model_name}-fp8"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
        return

    providers = ["RedHatAI", "arcee-ai", "neuralmagic", model_provider]

    target_repo_id = model_id

    try:
        # Search for available quantized versions linked to this base model
        # Also include a general search by name as a fallback pool
        models_linked = list(api.list_models(filter=model_id))
        models_search = list(api.list_models(search=model_name))

        # Combine and deduplicate
        all_candidates = {m.id: m for m in (models_linked + models_search)}.values()

        found = False

        # 1. Search for FP8-Dynamic models from preferred providers
        for provider in providers:
            for m in all_candidates:
                if (
                    m.id.lower().startswith(f"{provider.lower()}/")
                    and "fp8" in m.id.lower()
                    and "dynamic" in m.id.lower()
                ):
                    target_repo_id = m.id
                    found = True
                    break
                elif (
                    m.id.lower().startswith(f"{provider.lower()}/")
                    and "fp8" in m.id.lower()
                ):
                    target_repo_id = m.id
                    found = True
                    break
            if found:
                break

        if found:
            print(f"Found quantized model: {target_repo_id}")
            snapshot_download(repo_id=target_repo_id, local_dir=save_dir)
            print(f"Model {target_repo_id} downloaded to {save_dir}")
        else:
            print(f"No FP8 version found for {model_id}.")

    except Exception as e:
        print(f"Error searching for quantized models: {e}")


def quantize_model(model_id: str, cache_dir: Path | str = DEFAULT_CACHE) -> None:
    model_name = model_id.split("/")[-1]
    model_dir = Path(cache_dir) / model_name
    save_dir = Path(cache_dir) / f"{model_name}-fp8"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
        return

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Configure the quantization algorithm and scheme.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:.*mlp.gate$"],
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe, batch_size=1, num_calibration_samples=512)  # type: ignore

    # Confirm generations of the quantized model look sane.
    # print("========== SAMPLE GENERATION ==============")
    # dispatch_model(model)
    # input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    #     model.device
    # )
    # output = model.generate(input_ids, max_new_tokens=20)  # type: ignore
    # print(tokenizer.decode(output[0]))
    # print("==========================================")

    # Save to disk in compressed-tensors format.
    print(f"Saving quantized model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
