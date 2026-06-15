import argparse
from vllm import LLM, SamplingParams
import transformers


def main():
    parser = argparse.ArgumentParser(description="Quickly test an LLM with a prompt.")

    # Arguments
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model folder"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt to send to the model"
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temp", type=float, default=1, help="Temperature for sampling"
    )

    args = parser.parse_args()

    print(f"--- Loading model from: {args.model} ---")

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.tokens,
        skip_special_tokens=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    print(f"--- Generating response ---")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(
        prompt,
        sampling_params,
    )

    # Print the output
    for output in outputs:
        print("\n" + "=" * 20)
        print(f"PROMPT: {prompt}")
        print("-" * 20)
        print(f"RESPONSE: {output.outputs[0].text}")
        print("=" * 20 + "\n")


if __name__ == "__main__":
    main()
