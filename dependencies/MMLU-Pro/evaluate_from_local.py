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
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from pathlib import Path

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
CACHE_DIR = ROOT_DIR / ".llms"

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

# Increased context length from 4096 to 8192
max_model_length = 8192
max_new_tokens = 2048
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
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    model_path = CACHE_DIR / f"{args.model.split("/")[-1]}-fp8"

    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=float(args.gpu_util),
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_length,
        trust_remote_code=True,
        enable_lora=True if args.lora_path else False,
        # Disable multi-modal capabilities by setting all limits to 0,
        # ensuring the model focuses solely on text processing for MMLU-Pro.
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        # Batch size of 1 for evaluation to ensure accurate measurement of inference time and resource usage.
        max_num_seqs=1,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop=["Question:"],
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
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


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(CURRENT_DIR / "cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[:k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


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


def batch_inference(llm, sampling_params, lora_request, inference_batch):
    start = time.time()
    outputs = llm.generate(
        inference_batch,
        sampling_params,
        lora_request=lora_request,
    )
    logging.info(
        str(len(inference_batch))
        + "size batch costing time: "
        + str(time.time() - start)
    )
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
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
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params, lora_request = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(
        llm, sampling_params, lora_request, inference_batches
    )
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info(
        "this batch accu is: {}, corr: {}, wrong: {}\n".format(
            str(accu), str(corr), str(wrong)
        )
    )

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()

    # --- Apply Statistical Sampling if the flag is passed ---
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
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(
            subject, model, tokenizer, val_df, test_df, output_path
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
    for k, v in sta_dict.items():
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
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.80")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora_path", "-lp", type=str, default=None)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run using a statistically relevant subset of the dataset.",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=8192,
        help="Maximum model length.",
    )

    args = parser.parse_args()

    max_model_length = args.max_model_length

    if args.lora_path and not os.path.exists(args.lora_path):
        raise ValueError(f"LoRA path {args.lora_path} does not exist.")
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
        level=logging.DEBUG,
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
