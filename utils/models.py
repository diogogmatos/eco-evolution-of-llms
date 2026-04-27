import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

torch.cuda.empty_cache()

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from huggingface_hub import HfApi, snapshot_download

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CACHE_DIR = ROOT_DIR / ".llms"


def get_model(model_id: str) -> None:
    # Try to download a quantized version of the model first, if it exists.
    try:
        download_quantized_model(model_id=model_id)
    # If no quantized version is found, download the full-precision model and quantize it locally.
    except:
        download_model(model_id=model_id)
        quantize_model(model_id=model_id)


def download_model(model_id: str) -> None:
    model_name = model_id.split("/")[-1]
    save_dir = CACHE_DIR / f"{model_name}"
    snapshot_download(repo_id=model_id, local_dir=save_dir)
    print(f"Model {model_id} downloaded to {save_dir}")


def download_quantized_model(model_id: str) -> None:
    api = HfApi()
    model_provider = model_id.split("/")[0]
    model_name = model_id.split("/")[-1]
    save_dir = CACHE_DIR / f"{model_name}-fp8"

    if save_dir.exists() and save_dir.is_dir():
        print(f"Quantized model already exists at {save_dir}")
        return

    providers = ["RedHatAI", "arcee-ai", "neuralmagic", model_provider]

    target_repo_id = model_id

    # Search for available quantized versions linked to this base model
    models_linked = list(api.list_models(filter=f"base_model:quantized:{model_id}"))

    # Combine and deduplicate
    all_candidates = {m.id: m for m in (models_linked)}.values()

    found = False

    # 1. Search for FP8-Dynamic models from preferred providers
    for provider in providers:
        for m in all_candidates:
            if (
                m.id.lower().startswith(f"{provider.lower()}/")
                and "-fp8" in m.id.lower()
                and "-dynamic" in m.id.lower()
            ):
                target_repo_id = m.id
                found = True
                break
            elif (
                m.id.lower().startswith(f"{provider.lower()}/")
                and "-fp8" in m.id.lower()
            ):
                target_repo_id = m.id
                found = True
                break
        if found:
            break

    # 2. If no FP8 models found from preferred providers, search for any FP8 model
    if not found:
        for m in all_candidates:
            if "-fp8" in m.id.lower() and "-dynamic" in m.id.lower():
                target_repo_id = m.id
                found = True
                break
            elif "-fp8" in m.id.lower():
                target_repo_id = m.id
                found = True
                break

    if found:
        print(f"Found quantized model: {target_repo_id}")
        snapshot_download(repo_id=target_repo_id, local_dir=save_dir)
        print(f"Model {target_repo_id} downloaded to {save_dir}")
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
