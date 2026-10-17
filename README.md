# Eco-Evolution of LLMs: Replication Package

This repository contains the experimental pipeline and the data of the Master's dissertation *The Evolution of Intelligence Efficiency in Local, Open Large Language Models: A Controlled Empirical Study* (University of Minho).

The study measures how the energy efficiency of open large language models (LLMs) has evolved across successive releases. It measures the energy consumption, execution time and accuracy of 47 open models, from 20 releases of six families (Llama, Qwen, Gemma, DeepSeek, Phi and Mistral) published between July 2023 and March 2026. All the models run on the same hardware, with the same inference engine ([vLLM](https://github.com/vllm-project/vllm)) and on the same benchmark ([MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)). Every inference is measured individually: the energy of the GPU through NVML, and the energy of the processors and of their memory through Intel RAPL.

The repository can be used to:

- **inspect the data** of the experiment: 202,740 inferences from 545 runs, each with its energy, duration, token counts and the answer given by the model;
- **reproduce the analysis** of the dissertation, from the recorded results to every table and chart;
- **execute the experiment again**, or **evaluate new models** with the same pipeline.

## Contents

The pipeline has four stages, each of which reads what the previous one produces:

1. **Model selection** (`model-selection/`). A notebook collects the models of the selected releases from the Hugging Face Hub, together with their metadata, and writes the pool of candidate models (`models.csv`) and the selected model runs (`selected_models.csv`).
2. **Orchestration** (`experiments/`, `experiment-runner/`). [Experiment Runner](https://github.com/S2-group/experiment-runner) builds the run table from the selected models, downloads the models, collects their sampling parameters in `experiments/model_configs.json`, and launches the runs one at a time.
3. **Evaluation** (`dependencies/MMLU-Pro/`). For each run, `evaluate_from_local.py` loads one model with vLLM, brings the hardware to the same temperature, and answers a fixed sample of 372 MMLU-Pro questions while measuring the energy of each inference.
4. **Post-processing and analysis** (`post-processing/`, `analysis/`). The recorded results are corrected, and the tables and charts of the analysis are built from them.

```text
eco-evolution-of-llms/
├── model-selection/        # stage 1: notebook and its output files
├── experiment-runner/      # code of the Experiment Runner framework
├── experiments/            # stage 2: configuration of each execution
├── dependencies/MMLU-Pro/  # stage 3: evaluation script, summary of a run
├── utils/                  # download and configuration of the models
├── results/                # outputs of each execution, one directory per run
├── post-processing/        # stage 4: corrections of the recorded data
├── analysis/               # stage 4: notebooks, tables and charts
└── requirements.txt        # dependencies and installation instructions
```

The `results/` directory holds the outputs of three executions:

| Directory | Configuration | Content |
|---|---|---|
| `results/main/` | `experiments/main/` | Main execution: 530 runs, 10 repetitions of each of the 53 model runs |
| `results/mistral-bf16/` | `experiments/mistral-bf16/` | Supplementary execution: 15 runs of the three Ministral 3 instruct models with 16-bit weights |
| `results/idle_consumption/` | `experiments/idle-consumption/` | Idle consumption of the server, measured over 30 seconds per run |

Each execution has a `run_table.csv`, with one row per run and its aggregated metrics, and one directory per run (`run_<n>_repetition_<r>/`). A run directory contains the per-prompt records (`<model>/CoT/all/<subject>.json`, one file per subject of the benchmark), the summary of the run (`run_data.json`), its duration (`runtime.txt`) and its logs. The results of the main execution as originally recorded, before the corrections applied by the post-processing notebook, are available at the `original-results` tag.

## Requirements

- **Linux** on x86_64. Experiment Runner does not work on Windows.
- **An NVIDIA GPU** with CUDA 11.8, and `nvidia-smi` available in the `PATH`.
- **An Intel processor with RAPL**, exposing its package and DRAM domains through the Linux power capping framework (`/sys/class/powercap`). On recent kernels, the energy counters can only be read by root. To run the experiment as another user, make them readable (the permission is reset when the machine restarts):

  ```bash
  sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
  ```

- **A Hugging Face account.** Some of the models, such as those of the Llama and Gemma families, are gated: accept their licenses on the Hugging Face Hub, and authenticate before downloading them, with `hf auth login`.

The experiment of the dissertation was executed on a server with two Intel Xeon Gold 5418Y processors and an NVIDIA L40S GPU (48 GB), running Ubuntu 22.04.5 LTS. Its results are only comparable with those of another execution if both are obtained in the same environment.

The post-processing and analysis notebooks do not need a GPU and can be run on any machine.

## Installation

The installation instructions are also kept at the top of `requirements.txt`.

### Dependencies

- CUDA 11.8 (https://developer.nvidia.com/cuda-11-8-0-download-archive)

  The following variables should be set in your shell environment to ensure proper CUDA usage. Change `<path_to_cuda_11.8_installation>` to the actual path where CUDA 11.8 is installed on your system:

  ```bash
  export CUDA_HOME=<path_to_cuda_11.8_installation>
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### Installation instructions

From the root of the repository, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm==0.19.1 --torch-backend=auto
uv pip install -r requirements.txt
```

The versions of the main dependencies are fixed in `requirements.txt`, and `freeze.txt` lists the exact version of every installed package (the output of `pip freeze`).

## Usage

All the commands below are run from the root of the repository, with the virtual environment activated (`source .venv/bin/activate`). The notebooks can be opened in any editor with Jupyter support, such as VS Code, using `.venv` as the kernel.

### 1. Selecting the models

`model-selection/model-selection.ipynb` collects the models from the Hugging Face Hub and writes `models.csv` and `selected_models.csv`. Its outputs are already in the repository, so this step is only needed to change the selection or to add new releases. It requires access to the Hugging Face Hub. When a value cannot be obtained automatically, the notebook asks for it and saves the answer in `manual_model_data.json`, so that each value is only requested once.

### 2. Running an experiment

An execution is launched by passing its configuration to Experiment Runner:

```bash
python experiment-runner/ experiments/main/RunnerConfig.py
```

| Configuration | Purpose |
|---|---|
| `experiments/main/` | Main execution: every selected model run, 10 times each, in a shuffled order |
| `experiments/mistral-bf16/` | The three Ministral 3 instruct models with 16-bit weights |
| `experiments/idle-consumption/` | Energy consumed by the idle server over 30 seconds, 10 times |
| `experiments/test-run/` | Quick check of the setup: one repetition of each model, on the history questions only |
| `experiments/runtime-approx/` | One repetition of each model, with the full sample of questions, used to estimate the total runtime of the experiment before the main execution |

Before the first run, the models are downloaded to `.llms/` and their recommended sampling parameters are added to `experiments/model_configs.json`. The entries already present in this file are never overwritten, so the manual corrections made to it are preserved. The results are written to `results/<name>/`, where `<name>` is the `name` attribute of the configuration.

An interrupted execution can be resumed by launching the same command again: Experiment Runner executes only the runs that are not yet marked as done in the run table. For the same reason, it refuses to start an execution whose results directory already holds a completed experiment. To execute the experiment again without overwriting the published results, change the `name` attribute of the configuration (or move the existing results directory).

### 3. Post-processing the results

`post-processing/post-processing.ipynb` turns the results of the main execution as they were recorded into the results used in the analysis: it reconstructs the CPU and DRAM energy lost to a wrong handling of the overflow of the RAPL counters, and recomputes the summary of every run and the run table. Its corrections are already applied to `results/main/`. The notebook can be run again on the current results, since it restores the recorded values before applying the corrections, and it only writes to disk when `APPLY` is set to `True`. Every corrected value is listed in `results/main/rapl_corrections.csv`.

### 4. Reproducing the analysis

The notebooks of `analysis/` are run in order:

1. `00-dataset.ipynb` builds the tables used by the other notebooks, in `analysis/data/`, from the results of the executions.
2. `01-overview.ipynb` to `06-discussion.ipynb` answer the research questions of the dissertation. Their charts are saved to `analysis/charts/`.

The paths of the results, the number of repetitions of each execution and the style of the charts are defined in `analysis/common.py`.

### Evaluating new models

The pipeline can evaluate models released after the study without any change to its logic:

1. Add the collection of the new release to the list of collections of the model selection notebook, and run it again. Since the limits of the size categories are computed from the whole pool, compare the new pool and categories with the previous ones.
2. Experiment Runner does not add runs to a completed experiment, since this would change its run table, and with it the numbering and the order of the runs whose results are already recorded. Either create a configuration of its own for the new models, by copying `experiments/main/RunnerConfig.py` and restricting it to them, as `experiments/mistral-bf16/RunnerConfig.py` does, or move the results directory of the main experiment elsewhere and run it again from the start with the complete list of models. In both cases, review the new entries of `experiments/model_configs.json` against the model cards.
3. Run the configuration in the same environment as the previous executions.
4. If the new models have a configuration of their own, add its results directory and its number of repetitions to `analysis/common.py`. Then run the analysis notebooks again.

## Experiment Runner

The orchestration is based on [Experiment Runner](https://github.com/S2-group/experiment-runner), a generic framework for the automatic execution of measurement-based experiments, whose code is included in `experiment-runner/`. The `examples/`, `test/` and `test-standalone/` directories also come from it. Its documentation is available in its [repository](https://github.com/S2-group/experiment-runner) and [wiki](https://github.com/S2-group/experiment-runner/wiki), and it is described in the following publication:

```bibtex
@article{SCICO_2025,
  title = {{Experiment {Runner}: a {Tool} for the {Automatic} {Orchestration} of {Experiments} {Targeting} {Software} {Systems}}},
  issn = {0167-6423},
  journal = {Science of Computer Programming},
  author = {Max Karsten and {Andrei Calin} Dragomir and Radu Apsan and Vincenzo Stoico and Ivano Malavolta},
  year = {2025},
  pages = {103415},
  volume = {1},
  url = {https://www.sciencedirect.com/science/article/pii/S0167642325001546},
  doi = {https://doi.org/10.1016/j.scico.2025.103415}
}
```

## Citation

If you use this replication package, please cite it as described in `CITATION.cff`:

```bibtex
@misc{matos2026eco,
  author       = {Matos, Diogo},
  title        = {{Eco-Evolution of LLMs: Replication Package}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/diogogmatos/eco-evolution-of-llms}}
}
```

## License

This repository is distributed under the MIT License (see `LICENSE`).
