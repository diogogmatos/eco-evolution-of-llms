import argparse
from vllm import LLM, SamplingParams


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
        "--temp", type=float, default=0.7, help="Temperature for sampling"
    )

    args = parser.parse_args()

    print(f"--- Loading model from: {args.model} ---")

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=args.temp, max_tokens=args.tokens, top_p=0.95
    )

    print(f"--- Generating response ---")
    outputs = llm.generate([args.prompt], sampling_params)

    # Print the output
    for output in outputs:
        print("\n" + "=" * 20)
        print(f"PROMPT: {args.prompt}")
        print("-" * 20)
        print(f"RESPONSE: {output.outputs[0].text}")
        print("=" * 20 + "\n")


if __name__ == "__main__":
    main()
