# Summary of a run (run_data.json), computed from its per-prompt records.
# Used by evaluate_from_local.py at the end of a run, and by post-processing/post-processing.ipynb
# to recompute the summaries of the runs of the main experiment.

import statistics


def summarize_run(records, model, accuracy, max_new_tokens):
    nr_of_inferences = len(records)

    gpu_joules = [r["inference_gpu_joules"] for r in records]
    generated_tokens = [r["generated_tokens"] for r in records]
    inference_times = [r["inference_time_seconds"] for r in records]

    total_inference_gpu_joules = sum(gpu_joules)
    total_inference_cpu_joules = sum(r["inference_cpu_joules"] for r in records)
    total_inference_dram_joules = sum(r["inference_dram_joules"] for r in records)
    total_inference_joules = sum(r["total_inference_joules"] for r in records)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in records)
    total_generated_tokens = sum(generated_tokens)
    total_tokens = sum(r["total_tokens"] for r in records)
    total_thinking_tokens = sum(r["thinking_tokens"] for r in records)
    total_inference_time = sum(inference_times)
    total_prefill_time = sum(r["prefill_time_seconds"] for r in records)
    total_decode_time = sum(r["decode_time_seconds"] for r in records)
    sum_gpu_temp = sum(r["gpu_temperature_celsius"] for r in records)
    sum_cpu_temp = sum(r["cpu_temperature_celsius"] for r in records)

    # The efficiency metrics use the GPU energy only. The token metrics count generated tokens
    # only, since processing the prompt accounts for a small part of the energy of an inference.
    return {
        "model": model,
        "avg_accuracy": accuracy,
        "accuracy_per_gpu_joule": accuracy
        / (total_inference_gpu_joules / nr_of_inferences),
        # Reciprocal of the above. Undefined when no question was answered correctly.
        "gpu_joules_per_correct_answer": (
            total_inference_gpu_joules / (accuracy * nr_of_inferences)
            if accuracy > 0
            else None
        ),
        "generated_tokens_per_gpu_joule": total_generated_tokens
        / total_inference_gpu_joules,
        "gpu_joules_per_generated_token": total_inference_gpu_joules
        / total_generated_tokens,
        "avg_gpu_joules_per_inference": total_inference_gpu_joules / nr_of_inferences,
        "median_gpu_joules_per_inference": statistics.median(gpu_joules),
        "total_inference_joules": total_inference_joules,
        "total_inference_cpu_joules": total_inference_cpu_joules,
        "total_inference_dram_joules": total_inference_dram_joules,
        "total_inference_gpu_joules": total_inference_gpu_joules,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_thinking_tokens": total_thinking_tokens,
        "avg_generated_tokens_per_inference": total_generated_tokens
        / nr_of_inferences,
        "median_generated_tokens_per_inference": statistics.median(generated_tokens),
        # Responses that were cut off because they reached the limit of generated tokens
        "truncated_responses": sum(1 for t in generated_tokens if t >= max_new_tokens),
        # Responses from which no answer could be extracted (scored with a random option)
        "unparsed_responses": sum(1 for r in records if not r["pred"]),
        "avg_prefill_tokens_per_second": total_prompt_tokens / total_prefill_time,
        "avg_decode_tokens_per_second": total_generated_tokens / total_decode_time,
        "avg_time_per_inference_seconds": total_inference_time / nr_of_inferences,
        "median_time_per_inference_seconds": statistics.median(inference_times),
        "total_inference_time_seconds": total_inference_time,
        "total_prefill_time_seconds": total_prefill_time,
        "total_decode_time_seconds": total_decode_time,
        "avg_total_power_watts": total_inference_joules / total_inference_time,
        "avg_cpu_power_watts": total_inference_cpu_joules / total_inference_time,
        "avg_dram_power_watts": total_inference_dram_joules / total_inference_time,
        "avg_gpu_power_watts": total_inference_gpu_joules / total_inference_time,
        "avg_gpu_temperature_celsius": sum_gpu_temp / nr_of_inferences,
        "avg_cpu_temperature_celsius": sum_cpu_temp / nr_of_inferences,
    }
