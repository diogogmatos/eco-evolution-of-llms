from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def quantize_model(model_id: str, cache_dir: str = ".llms") -> None:
    # Load model.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Configure the quantization algorithm and scheme.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:.*mlp.gate$"],
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe) # type: ignore

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")

    # Save to disk in compressed-tensors format.
    # Convert model_id to a safe directory name (e.g., "google/gemma-2b" -> "gemma-2b-fp8")
    model_name = model_id.split("/")[-1]
    save_dir = f"{model_name}-fp8"

    print(f"Saving quantized model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
