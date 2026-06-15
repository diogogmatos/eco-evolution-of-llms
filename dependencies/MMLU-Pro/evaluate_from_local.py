# This code was adapted from TIGER-AI-Lab/MMLU-Pro/evaluate_from_local.py

import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
import math
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from pathlib import Path
import pynvml
from utils import (
    get_gpu_temp,
    get_cpu_temp,
    cpu_burn_step,
    async_cpu_burn_step,
    load_mistral_system_prompt,
)

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
cpu_rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
dram_rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj"


def get_status():
    # with open(cpu_rapl_path, "r") as f:
    #     cpu_uj = int(f.read().strip())

    # with open(dram_rapl_path, "r") as f:
    #     dram_uj = int(f.read().strip())

    gpu_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    return 0, 0, gpu_mj


CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
CACHE_DIR = ROOT_DIR / ".llms"

BASELINE_TEMP_GPU = 55  # Target temperature in Celsius to ensure consistent performance
BASELINE_TEMP_CPU = 35  # Target temperature in Celsius to ensure consistent performance
GPU_TOLERANCE = 1
CPU_TOLERANCE = 2

choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]

random.seed(12345)


# Use stratified sampling to select a statistically relevant subset of the test set.
def apply_stratified_sampling(test_df):
    N = len(test_df)
    if N == 0:
        return test_df

    # Standard statistical parameters
    z = 1.96  # 95% confidence level
    p = 0.5  # 50% estimated proportion (most conservative)
    e = 0.05  # 5% margin of error

    # Infinite population sample size
    n0 = (z**2 * p * (1 - p)) / (e**2)
    # Finite population correction
    n = math.ceil(n0 / (1 + (n0 - 1) / N))

    logging.info("--- Stratified Sampling Activated ---")
    logging.info(f"Original Population Size (N): {N}")
    logging.info(f"Calculated Target Sample Size (n): {n}")

    # Group questions by category
    categories = {}
    for item in test_df:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    sampled_df = []
    # Calculate proportion and sample for each category
    for cat, items in categories.items():
        Nh = len(items)
        # Calculate proportional size for this category
        nh = round((Nh / N) * n)

        # Shuffle deterministically (uses the global random.seed)
        random.shuffle(items)
        sampled_df.extend(items[:nh])

        logging.info(f"Category '{cat}': Population {Nh} -> Sampled {nh}")

    logging.info(f"Total Sampled Size (after rounding): {len(sampled_df)}")
    logging.info("-------------------------------------")

    return sampled_df


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(ROOT_DIR / ".datasets"))
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model_config(config_path, model_name):
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise ValueError(f"Config path {config_path} does not exist.")

    with open(config_file, "r", encoding="utf-8") as fi:
        config_data = json.load(fi)

    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a JSON object.")

    if "models" in config_data and isinstance(config_data["models"], dict):
        model_configs = config_data["models"]
        resolved = dict(config_data.get("default", {}))
    else:
        model_configs = config_data
        resolved = (
            dict(config_data.get("default", {}))
            if isinstance(config_data.get("default"), dict)
            else {}
        )

    model_full_name = model_name
    if args.enable_thinking is not None and args.enable_thinking:
        model_full_name += " (Thinking Mode)"
    elif args.enable_thinking is not None and not args.enable_thinking:
        model_full_name += " (Instruct Mode)"

    model_config = model_configs.get(model_full_name)
    if isinstance(model_config, dict):
        resolved.update(model_config)

    logging.info(f"Resolved configuration for model '{model_full_name}': {resolved}")

    return resolved


def load_model():
    model_config = load_model_config(args.config_path, args.model)
    max_model_length = int(model_config.get("max_model_length", args.max_model_length))
    max_new_tokens = int(model_config.get("max_new_tokens", args.max_new_tokens))
    max_thinking_tokens = int(
        model_config.get("max_thinking_tokens", args.max_thinking_tokens)
    )

    model_path = Path(
        model_config.get("model_path", CACHE_DIR / args.model.split("/")[-1])
    )
    gpu_memory_utilization = float(model_config.get("gpu_util", args.gpu_util))

    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_lora=True if args.lora_path else False,
        max_model_len=max_model_length,
        # Batch size of 1 for evaluation to ensure accurate measurement of inference time and resource usage.
        max_num_seqs=1,
        # Use CUDA Graphs for an optimized engine environment.
        enforce_eager=False,
        # Disable vLLM's internal prefix caching optimizations which can interfere with readings.
        enable_prefix_caching=False,
        # Enable vLLM internal logging for detailed metrics.
        disable_log_stats=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    stop_strings = ["USER:", "ASSISTANT:"]
    # Only use question-based stop tokens for instruct models
    if args.max_thinking_tokens == 0:
        stop_strings.extend(["Question:", "\nQuestion"])
    # Dynamically resolve stop strings and token ids
    extra_stop_strings = model_config.get("stop_strings", [])
    if isinstance(extra_stop_strings, list):
        for stop_string in extra_stop_strings:
            if stop_string not in stop_strings:
                stop_strings.append(stop_string)

    stop_token_ids = []
    if isinstance(tokenizer.eos_token_id, list):
        stop_token_ids.extend(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Safely catch conversational End-Of-Turn tokens (e.g. <|eot_id|>, <|im_end|>)
    for special_token in getattr(tokenizer, "all_special_tokens", []):
        token_lower = special_token.lower()

        # Ensure we don't accidentally catch end_thought tokens
        if "thought" in token_lower or "reason" in token_lower:
            continue

        if any(keyword in special_token.lower() for keyword in ["eos", "end", "eot"]):
            # Convert the special token string to its integer ID
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if token_id is not None and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)

            # Keep string fallback for safety
            if special_token not in stop_strings:
                stop_strings.append(special_token)

    extra_stop_token_ids = model_config.get("stop_token_ids", [])
    if isinstance(extra_stop_token_ids, list):
        for token_id in extra_stop_token_ids:
            if token_id not in stop_token_ids:
                stop_token_ids.append(token_id)

    stop_token_ids = list(set(stop_token_ids))
    # Ensure max_new_tokens accounts for the thinking token budget
    # to ensure thinking models have the same number of tokens to provide the final answer.
    max_new_tokens = max_thinking_tokens + max_new_tokens

    sampling_params = SamplingParams(
        temperature=model_config.get("temperature", 1.0),
        top_p=model_config.get("top_p", 1.0),
        top_k=model_config.get("top_k", 0.0),
        min_p=model_config.get("min_p", 0.0),
        repetition_penalty=model_config.get("repetition_penalty", 1.0),
        presence_penalty=model_config.get("presence_penalty", 0.0),
        frequency_penalty=model_config.get("frequency_penalty", 0.0),
        max_tokens=max_new_tokens,
        stop=stop_strings,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(
            lora_name="lora", lora_path=args.lora_path, lora_int_id=1
        )

    return (llm, sampling_params, lora_request), tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."

    return prompt


def has_system_role(tokenizer):
    if getattr(tokenizer, "chat_template", None) is not None:
        test_chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]

        try:
            _ = tokenizer.apply_chat_template(test_chat, tokenize=False)
            return True
        except:
            return False

    return False


def generate_cot_prompt(val_df, curr, k, tokenizer, model_config):
    is_thinking_model = args.max_thinking_tokens > 0
    prompt = ""
    with open(CURRENT_DIR / "cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[:k]
    prompt = prompt.replace("{$}", subject) + "\n"

    if getattr(tokenizer, "chat_template", None) is not None:
        messages = []
        system_prompt = prompt + model_config.get("additional_system_prompt", "")

        for example in val_df:
            user_content = "Question:\n" + example["question"] + "\nOptions:\n"
            for i, opt in enumerate(example["options"]):
                user_content += "{}. {}\n".format(choices[i], opt)

            if len(messages) == 0:
                if has_system_role(tokenizer):
                    messages.append(
                        {"role": "system", "content": system_prompt}
                        if "mistralai" not in args.model
                        else load_mistral_system_prompt(system_prompt)
                    )
                else:
                    user_content = system_prompt + user_content

            messages.append({"role": "user", "content": user_content})

            assistant_content = example["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            messages.append({"role": "assistant", "content": assistant_content})

        user_content = "Question:\n" + curr["question"] + "\nOptions:\n"
        for i, opt in enumerate(curr["options"]):
            user_content += "{}. {}\n".format(choices[i], opt)

        if len(messages) == 0:
            if has_system_role(tokenizer):
                messages.append({"role": "system", "content": system_prompt})
            else:
                user_content = system_prompt + user_content

        messages.append({"role": "user", "content": user_content})

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if args.enable_thinking is not None:
            template_kwargs["enable_thinking"] = args.enable_thinking

        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

        if model_config.get("additional_suffix", "") != "":
            prompt += model_config["additional_suffix"]

        # Do not append suffix for reasoning models as it interferes with the model's internal reasoning process
        if not is_thinking_model:
            # Check if the model uses common delimiters and safely handle the suffix placement
            if prompt.endswith(" ") or prompt.endswith("\n"):
                prompt += "Answer: Let's think step by step."
            else:
                # Ensure there is a distinct spacer or clean transition into the text generation space
                prompt += " Answer: Let's think step by step."

        if args.debug:
            print("PROMPT\n" + prompt + "\n")
    else:
        # Fallback for old base models without chat templates
        for example in val_df:
            prompt += format_cot_example(example, including_answer=True)
        prompt += format_cot_example(curr, including_answer=False)

    return prompt


def extract_answer(text):
    # Remove reasoning/thinking blocks to avoid false positives in answer extraction
    thinking_pattern = r"(?:(?:<think>|<reasoning>|<\|channel>thought|\[THINK\])?.*?(?:</think>|</reasoning>|<channel\|>|\[/THINK\]))"
    clean_text = re.sub(thinking_pattern, "", text, flags=re.DOTALL).strip()
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, clean_text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + clean_text)
        return extract_again(clean_text)


def extract_again(text):
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def prepare_prompts(subject, tokenizer, val_df, test_df):
    global choices
    logging.info("Preparing prompts for " + subject)
    inference_batches = []
    model_config = load_model_config(args.config_path, args.model)

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k, tokenizer, model_config)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < args.max_model_length - args.max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)
    return inference_batches


def count_thinking_tokens(reasoning_text, tokenizer):
    if not reasoning_text or reasoning_text.strip() == "":
        return 0
    # Encode the raw text to token IDs without adding special system tokens (like BOS)
    thinking_token_ids = tokenizer.encode(reasoning_text, add_special_tokens=False)
    return len(thinking_token_ids)


def calculate_inference_metrics(
    warmup_metrics: list[Metric], accumulated_metrics: list[Metric]
) -> list[Metric]:
    for wmetric, ametric in zip(warmup_metrics, accumulated_metrics):
        if isinstance(wmetric, Gauge) and isinstance(ametric, Gauge):
            ametric.value -= wmetric.value
        elif isinstance(wmetric, Counter) and isinstance(ametric, Counter):
            ametric.value -= wmetric.value
        elif isinstance(wmetric, Histogram) and isinstance(ametric, Histogram):
            ametric.count -= wmetric.count
            ametric.sum -= wmetric.sum

    return accumulated_metrics


def get_metric_value(metrics: list[Metric], name: str):
    for metric in metrics:
        if metric.name == name:
            if isinstance(metric, Gauge):
                return metric.value
            elif isinstance(metric, Counter):
                return metric.value
            elif isinstance(metric, Histogram):
                return metric.sum
            else:
                raise ValueError(f"Unsupported metric type: {type(metric)}")


def main():
    model, tokenizer = load_model()
    llm, sampling_params, lora_request = model

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()

    # Apply Statistical Sampling if the flag is passed
    if args.sample:
        full_test_df = apply_stratified_sampling(full_test_df)

    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), "a") as f:
        f.write("\n------category level sta------\n")

    # PHASE 1: Prepare all prompts in advance to exclude CPU overhead from energy readings
    subject_prepared_data = {}
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df_sub = select_by_category(full_test_df, subject)
        val_df_sub = select_by_category(full_val_df, subject)

        inference_batches = prepare_prompts(subject, tokenizer, val_df_sub, test_df_sub)

        subject_prepared_data[subject] = {
            "test_df": test_df_sub,
            "inference_batches": inference_batches,
        }

    # PHASE 2: Run inference sequentially but back-to-back

    # Flatten prompts and track origins to eliminate data management overhead during timing
    flat_inference_prompts = []
    flat_prompt_lengths = []
    prompt_metadata = []  # Keeps track of (subject, array_index) pairs

    for subject in selected_subjects:
        batches = subject_prepared_data[subject]["inference_batches"]
        num_items = len(batches)

        # Pre-allocate specific array lists to prevent runtime dynamic allocations later
        subject_prepared_data[subject]["pred_batch"] = [None] * num_items
        subject_prepared_data[subject]["response_batch"] = [None] * num_items
        subject_prepared_data[subject]["metrics_batch"] = [None] * num_items

        for idx, prompt in enumerate(batches):
            flat_inference_prompts.append(prompt)
            prompt_ids = tokenizer(prompt)["input_ids"]
            flat_prompt_lengths.append(len(prompt_ids))
            prompt_metadata.append((subject, idx))

    # 95th percentile prompt
    sorted_lengths = sorted(flat_prompt_lengths)
    idx = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
    p95_length = sorted_lengths[idx]
    warmup_prompt = next(
        prompt
        for prompt, length in zip(flat_inference_prompts, flat_prompt_lengths)
        if length >= p95_length
    )

    # Warmup to initialize driver states, kernels, vLLM KV Cache blocks and heat system up.
    logging.info("Starting warmup...")

    gpu_now = get_gpu_temp()
    cpu_now = get_cpu_temp()

    logging.info(f"[Current] GPU: {gpu_now}°C | CPU: {cpu_now}°C")
    logging.info(
        f"[Target] GPU: {BASELINE_TEMP_GPU}°C +/- {GPU_TOLERANCE}°C | CPU: {BASELINE_TEMP_CPU}°C +/- {CPU_TOLERANCE}°C)"
    )

    # Run the warmup prompt multiple times to ensure all components are fully initialized and optimized
    llm.generate(
        [warmup_prompt for _ in range(5)],
        sampling_params,
        lora_request=lora_request,
    )

    gpu_increase_per_prompt = (get_gpu_temp() - gpu_now) / 5

    # Run additional burn-in workloads if the system is still bellow baseline
    while True:
        gpu_temp = get_gpu_temp()
        cpu_temp = get_cpu_temp()
        needed_gpu_increase = BASELINE_TEMP_GPU - gpu_temp
        needed_cpu_increase = BASELINE_TEMP_CPU - cpu_temp

        logging.info(f"[Current] GPU: {gpu_temp}°C | CPU: {cpu_temp}°C")
        logging.info(
            f"[Target] GPU: {BASELINE_TEMP_GPU}°C +/- {GPU_TOLERANCE}°C | CPU: {BASELINE_TEMP_CPU}°C +/- {CPU_TOLERANCE}°C)"
        )

        gpu_ready = (
            (BASELINE_TEMP_GPU - GPU_TOLERANCE)
            <= gpu_temp
            <= (BASELINE_TEMP_GPU + GPU_TOLERANCE)
        )
        cpu_ready = (
            (BASELINE_TEMP_CPU - CPU_TOLERANCE)
            <= cpu_temp
            <= (BASELINE_TEMP_CPU + CPU_TOLERANCE)
        )

        if gpu_ready and cpu_ready:
            logging.info("Thermal baseline reached.")
            logging.info("Warmup completed, starting inference...")
            break

        if (
            gpu_temp < BASELINE_TEMP_GPU - GPU_TOLERANCE
            and cpu_temp < BASELINE_TEMP_CPU - CPU_TOLERANCE
        ):
            logging.info(
                "GPU and CPU are below baseline temperature. Continuing warmup..."
            )

            # Each burn step should increase CPU temp by ~2°C every 5 seconds
            cpu_burn_duration = (max(needed_cpu_increase // 2, 1)) * 5
            t_start = time.time()
            async_cpu_burn_step(cpu_burn_duration)

            llm.generate(
                [
                    warmup_prompt
                    for _ in range(
                        max(
                            1,
                            math.ceil(needed_gpu_increase / gpu_increase_per_prompt)
                            + 2,
                        )
                    )
                ],
                sampling_params,
                lora_request=lora_request,
            )
            t_stop = time.time()

            # Wait for CPU burn to complete
            if (t_stop - t_start) < cpu_burn_duration:
                time.sleep(cpu_burn_duration - (t_stop - t_start))

        if (
            gpu_temp < BASELINE_TEMP_GPU - GPU_TOLERANCE
            and cpu_temp >= BASELINE_TEMP_CPU - CPU_TOLERANCE
        ):
            logging.info("GPU is below baseline temperature. Continuing warmup...")
            llm.generate(
                [
                    warmup_prompt
                    for _ in range(
                        max(
                            1,
                            math.ceil(needed_gpu_increase / gpu_increase_per_prompt)
                            + 2,
                        )
                    )
                ],
                sampling_params,
                lora_request=lora_request,
            )

        if (
            cpu_temp < BASELINE_TEMP_CPU - CPU_TOLERANCE
            and gpu_temp >= BASELINE_TEMP_GPU - GPU_TOLERANCE
        ):
            logging.info("CPU is below baseline temperature. Continuing warmup...")
            for _ in range(
                max(1, needed_cpu_increase // 2)
            ):  # Each burn step should increase CPU temp by ~2°C
                cpu_burn_step()

        gpu_too_hot = gpu_temp > (BASELINE_TEMP_GPU + GPU_TOLERANCE)
        cpu_too_hot = cpu_temp > (BASELINE_TEMP_CPU + CPU_TOLERANCE)
        gpu_too_cold = gpu_temp < (BASELINE_TEMP_GPU - GPU_TOLERANCE)
        cpu_too_cold = cpu_temp < (BASELINE_TEMP_CPU - CPU_TOLERANCE)

        if gpu_too_hot or cpu_too_hot:
            logging.info("Entering cooldown...")

            if gpu_too_hot and cpu_too_hot:
                logging.info("CPU and GPU above baseline. Resting...")
                time.sleep(1)

            elif gpu_too_hot and cpu_too_cold:
                logging.info(
                    "GPU above baseline. CPU below baseline. Resting GPU, pulsing CPU..."
                )
                cpu_burn_step(2)

            elif cpu_too_hot and gpu_too_cold:
                logging.info(
                    "CPU above baseline. GPU below baseline. Resting CPU, pulsing GPU with warmup prompt..."
                )
                llm.generate(
                    warmup_prompt,
                    sampling_params,
                    lora_request=lora_request,
                )

            else:
                logging.info(
                    "One component overheating, the other is stable. Resting..."
                )
                time.sleep(1)

    flat_outputs = []

    warmup_metrics = llm.get_metrics()

    torch.cuda.synchronize()
    inference_start = time.time()
    cpu_start, dram_start, gpu_start = get_status()

    # Process prompts. vLLM will natively process these sequentially due to max_num_seqs=1
    flat_outputs = llm.generate(
        flat_inference_prompts, sampling_params, lora_request=lora_request
    )

    torch.cuda.synchronize()
    inference_end = time.time()
    cpu_end, dram_end, gpu_end = get_status()

    inference_metrics = calculate_inference_metrics(warmup_metrics, llm.get_metrics())

    for metric in inference_metrics:
        print(metric)

    # Post-processing Block: Extract tokens and evaluate strings outside the timed window
    for (subject, idx), output in zip(prompt_metadata, flat_outputs):
        generated_text = output.outputs[0].text
        thinking_pattern = r"(?:(?:<think>|<reasoning>|<\|channel>thought|\[THINK\])?.*?(?:</think>|</reasoning>|<channel\|>|\[/THINK\]))"
        think_match = re.search(thinking_pattern, generated_text, re.DOTALL)
        thinking_text = think_match.group(0).strip() if think_match else ""
        pred = extract_answer(generated_text)

        # Structure payload back into the subject map exactly as Phase 3 expects it
        subject_prepared_data[subject]["pred_batch"][idx] = pred
        subject_prepared_data[subject]["response_batch"][idx] = generated_text
        subject_prepared_data[subject]["metrics_batch"][idx] = {
            "prompt_tokens": (
                len(output.prompt_token_ids)
                if output.prompt_token_ids is not None
                else 0
            ),
            "generated_tokens": (
                len(output.outputs[0].token_ids)
                if output.outputs[0].token_ids is not None
                else 0
            ),
            "thinking_tokens": count_thinking_tokens(thinking_text, tokenizer),
        }

    # Phase 3: Calculate metrics and save results
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_thinking_tokens = 0

    for subject in selected_subjects:
        test_df_sub = subject_prepared_data[subject]["test_df"]
        pred_batch = subject_prepared_data[subject]["pred_batch"]
        response_batch = subject_prepared_data[subject]["response_batch"]
        metrics_batch = subject_prepared_data[subject]["metrics_batch"]
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))

        res = []
        prompt_tokens_sub = 0
        generated_tokens_sub = 0
        generated_thinking_tokens_sub = 0
        for j, curr in enumerate(test_df_sub):
            curr["pred"] = pred_batch[j]
            curr["model_outputs"] = response_batch[j]
            curr["prompt_tokens"] = metrics_batch[j]["prompt_tokens"]
            curr["generated_tokens"] = metrics_batch[j]["generated_tokens"]
            curr["thinking_tokens"] = metrics_batch[j]["thinking_tokens"]
            prompt_tokens_sub += curr["prompt_tokens"]
            generated_tokens_sub += curr["generated_tokens"]
            generated_thinking_tokens_sub += curr["thinking_tokens"]
            res.append(curr)

        total_prompt_tokens += prompt_tokens_sub
        total_generated_tokens += generated_tokens_sub
        total_thinking_tokens += generated_thinking_tokens_sub

        acc, corr_count, wrong_count = save_res(res, output_path)
        logging.info(
            "this batch accu is: {}, corr: {}, wrong: {}\n".format(
                str(acc), str(corr_count), str(wrong_count)
            )
        )

        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), "a") as f:
            f.write(
                "Average accuracy {:.4f} - {}\n".format(
                    sta_dict[subject]["accu"], subject
                )
            )

    total_corr, total_wrong = 0.0, 0.0
    for _, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), "a") as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, "a", newline="") as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)

    timestamp_path = os.path.join(args.save_dir, "run_data.json")
    with open(timestamp_path, "w") as f:
        inference_time_seconds = inference_end - inference_start
        inference_cpu_joules = (cpu_end - cpu_start) / 1e6
        inference_dram_joules = (dram_end - dram_start) / 1e6
        inference_gpu_joules = (gpu_end - gpu_start) / 1000.0
        total_inference_joules = (
            inference_cpu_joules + inference_dram_joules + inference_gpu_joules
        )
        total_tokens = total_prompt_tokens + total_generated_tokens
        nr_of_inferences = len(flat_inference_prompts)
        json.dump(
            {
                "model": (
                    args.model + " (Instruct Mode)"
                    if args.enable_thinking is not None and not args.enable_thinking
                    else (
                        args.model + " (Thinking Mode)"
                        if args.enable_thinking is not None and args.enable_thinking
                        else args.model
                    )
                ),
                "avg_accuracy": weighted_acc,
                "accuracy_per_joule": weighted_acc / total_inference_joules,
                "tokens_per_joule": total_tokens / total_inference_joules,
                "joules_per_token": total_inference_joules / total_tokens,
                "total_inference_joules": total_inference_joules,
                "total_tokens": total_tokens,
                "avg_tokens_per_second": 1
                / (
                    (
                        get_metric_value(
                            inference_metrics,
                            "vllm:request_time_per_output_token_seconds",
                        )
                        or 0
                    )
                    / nr_of_inferences
                ),
                "avg_time_per_inference_seconds": inference_time_seconds
                / nr_of_inferences,
                "total_inference_time_seconds": inference_time_seconds,
                "avg_total_power_watts": total_inference_joules
                / inference_time_seconds,
                "avg_time_to_first_token_seconds": (
                    get_metric_value(
                        inference_metrics, "vllm:time_to_first_token_seconds"
                    )
                    or 0
                )
                / nr_of_inferences,
                "total_prefill_time_seconds": get_metric_value(
                    inference_metrics, "vllm:request_prefill_time_seconds"
                )
                or 0,
                "total_decode_time_seconds": get_metric_value(
                    inference_metrics, "vllm:request_decode_time_seconds"
                )
                or 0,
                "inference_cpu_joules": inference_cpu_joules,
                "inference_dram_joules": inference_dram_joules,
                "inference_gpu_joules": inference_gpu_joules,
                "avg_cpu_power_watts": inference_cpu_joules / inference_time_seconds,
                "avg_dram_power_watts": inference_dram_joules / inference_time_seconds,
                "avg_gpu_power_watts": inference_gpu_joules / inference_time_seconds,
                "total_prompt_tokens": total_prompt_tokens,
                "total_generated_tokens": total_generated_tokens,
                "total_thinking_tokens": total_thinking_tokens,
            },
            f,
            indent=4,
        )

    # Cleanup vLLM process group to prevent resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--global_record_file",
        "-grf",
        type=str,
        default=str(ROOT_DIR / "results/eval_record_collection.csv"),
    )
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.90")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora_path", "-lp", type=str, default=None)
    parser.add_argument(
        "--config_path",
        "-cp",
        type=str,
        default=None,
        help="Path to a JSON config file with per-model overrides.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run using a statistically relevant subset of the dataset.",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=4096,
        help="Maximum model length.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens the model can generate in its response, includes thinking tokens.",
    )
    parser.add_argument(
        "--max_thinking_tokens",
        type=int,
        default=0,
        help="Maximum number of tokens allocated for the model's 'thinking' process before it starts generating the final answer.",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Whether to enable or disable the 'thinking' process for models that support it. If not specified, it will be assumed that the model doesn't support the feature and will be ran with its default behavior.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size and fewer inference iterations.",
    )

    args = parser.parse_args()

    if args.lora_path and not os.path.exists(args.lora_path):
        raise ValueError(f"LoRA path {args.lora_path} does not exist.")
    if args.config_path and not os.path.exists(args.config_path):
        raise ValueError(f"Config path {args.config_path} does not exist.")
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(args.save_dir, "/".join(args_generate_path(args)))
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime("%m-%d_%H-%M", time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    save_log_dir, file_name.replace("_summary.txt", "_logfile.log")
                )
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    main()
