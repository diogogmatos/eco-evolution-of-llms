"""
CPU power profile experiment.

Measures the power drawn by both CPU packages of the server, and by the DRAM attached to each of
them, while the system is idle and while a model is being evaluated.

The main experiment only measured the first CPU package. The results of this experiment are used
to estimate the energy of the second package (see energy-corrections/cpu-energy-corrections.ipynb).

Every run samples all Intel RAPL domains at a fixed interval:
  - "idle" workload: the system is left idle for IDLE_DURATION_SECONDS.
  - model workloads: the model is evaluated as in the main experiment, but on a single subject of the
    benchmark, and only the samples taken while the evaluation script is performing inference are considered.

The samples of each run are saved to `power_samples.csv` in the run directory.
"""

import csv
import glob
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from os.path import dirname, realpath
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.OperationType import OperationType
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from EventManager.EventSubscriptionController import EventSubscriptionController
from EventManager.Models.RunnerEvents import RunnerEvents
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

ROOT_DIR = Path(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(str(ROOT_DIR))

from utils.models import download_model

RAPL_DIR = "/sys/class/powercap"

# Workloads to profile: the idle system, plus models of different sizes and operating modes
# (names as they appear in model-selection/selected_models.csv).
IDLE = "idle"
WORKLOADS = [
    IDLE,
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Ministral-3-8B-Reasoning-2512",
]

# Subject of MMLU-Pro used for the model workloads. The power of the processors does not depend on the
# questions, so the smallest subject is enough and keeps the experiment short.
SELECTED_SUBJECT = "history"

IDLE_DURATION_SECONDS = 300
# The inference phase of a small model on a single subject only lasts a few seconds
SAMPLE_INTERVAL_SECONDS = 1

# Messages logged by evaluate_from_local.py that delimit the inference phase
INFERENCE_START_MESSAGE = "Warmup completed, starting inference..."
INFERENCE_END_MESSAGE = "Inference completed."


def find_rapl_domains():
    """Returns {column name: path} for every CPU package (cpu0, cpu1, ...) and its DRAM (dram0, dram1, ...)."""
    domains = {}
    for path in sorted(glob.glob(os.path.join(RAPL_DIR, "intel-rapl:*"))):
        with open(os.path.join(path, "name"), "r") as f:
            name = f.read().strip()
        package_index = os.path.basename(path).split(":")[1]
        if name.startswith("package"):
            domains[f"cpu{package_index}"] = path
        elif name == "dram":
            domains[f"dram{package_index}"] = path
    return domains


def read_int(path):
    with open(path, "r") as f:
        return int(f.read().strip())


class PowerSampler(threading.Thread):
    """Samples the average power of every RAPL domain at a fixed interval, in a background thread."""

    def __init__(self, domains, interval):
        super().__init__(daemon=True)
        self.domains = domains
        self.interval = interval
        self.max_ranges = {name: read_int(os.path.join(path, "max_energy_range_uj")) for name, path in domains.items()}
        self.samples = []
        self._stop_event = threading.Event()

    def _read(self):
        return {name: read_int(os.path.join(path, "energy_uj")) for name, path in self.domains.items()}

    def run(self):
        previous, previous_time = self._read(), time.time()
        while not self._stop_event.wait(self.interval):
            current, current_time = self._read(), time.time()
            sample = {"start_time": previous_time, "end_time": current_time}
            for name in self.domains:
                delta = current[name] - previous[name]
                # The counter restarted from zero during the interval
                if delta < 0:
                    delta += self.max_ranges[name]
                sample[f"{name}_watts"] = delta / 1e6 / (current_time - previous_time)
            self.samples.append(sample)
            previous, previous_time = current, current_time

    def stop(self):
        self._stop_event.set()
        self.join()


def find_log_time(log_path, message):
    """Returns the time at which the evaluation script logged the given message, or None."""
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if message in line:
                try:
                    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()
                except ValueError:
                    continue
    return None


class RunnerConfig:
    CURRENT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name: str = "cpu_power_profile"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path: Path = ROOT_DIR / "results"

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type: OperationType = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms: int = 60000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events(
            [
                (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
                (RunnerEvents.BEFORE_RUN, self.before_run),
                (RunnerEvents.START_RUN, self.start_run),
                (RunnerEvents.START_MEASUREMENT, self.start_measurement),
                (RunnerEvents.INTERACT, self.interact),
                (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
                (RunnerEvents.STOP_RUN, self.stop_run),
                (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
                (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
            ]
        )
        self.run_table_model = None  # Initialized later
        self.process = None
        self.sampler = None

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        workload = FactorModel("workload", WORKLOADS)
        self.run_table_model = RunTableModel(
            factors=[workload],
            exclude_combinations=[],
            data_columns=[
                "phase",
                "duration_seconds",
                "samples",
                "cpu0_watts",
                "cpu1_watts",
                "dram0_watts",
                "dram1_watts",
                "cpu_busy_watts",
                "cpu_other_watts",
                "package0_busy_share",
            ],
            repetitions=1,
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""

        self.env = os.environ.copy()
        self.models = pd.read_csv(ROOT_DIR / "model-selection/selected_models.csv")
        self.domains = find_rapl_domains()

        missing = {"cpu0", "cpu1", "dram0", "dram1"} - set(self.domains)
        if missing:
            raise RuntimeError(f"RAPL domains not found: {sorted(missing)}. Found: {sorted(self.domains)}")

        for workload in WORKLOADS:
            if workload != IDLE:
                download_model(workload.split(" ")[0])

        output.console_log("Config.before_experiment() called!")

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""

        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""

        workload = context.execute_run["workload"]
        self.process = None

        if workload != IDLE:
            # Evaluate the model as in the main experiment (experiments/main/RunnerConfig.py), on a single subject
            model = self.models[self.models["model"] == workload].iloc[0]
            max_model_length = int(model["max_model_length"])
            is_thinking = bool(model["is_thinking"])
            supports_enable_thinking = bool(model["supports_enable_thinking"])

            eval_script = ROOT_DIR / "dependencies/MMLU-Pro/evaluate_from_local.py"
            cmd = f"python {eval_script} --model {workload.split(' ')[0]} --save_dir {context.run_dir} --global_record_file {Path(dirname(context.run_dir)) / 'eval_record_collection.csv'} --config_path {ROOT_DIR / 'experiments/model_configs.json'} --max_model_length {min(max_model_length, 10240)} --max_thinking_tokens {4096 if is_thinking else 0} {f'--enable_thinking {is_thinking}' if supports_enable_thinking else ''} --ntrain {0 if is_thinking else 5} --selected_subjects {SELECTED_SUBJECT} --sample"

            output.console_log(f"Running command: {cmd}")
            self.log = open(context.run_dir / "experiment.log", "w")
            self.process = subprocess.Popen(cmd, shell=True, stdout=self.log, stderr=self.log, env=self.env)

        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""

        self.sampler = PowerSampler(self.domains, SAMPLE_INTERVAL_SECONDS)
        self.sampler.start()

        output.console_log("Config.start_measurement() called!")

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""

        if self.process is None:
            time.sleep(IDLE_DURATION_SECONDS)
        else:
            self.process.wait()
            self.log.close()

        output.console_log("Config.interact() called!")

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""

        self.sampler.stop()
        workload = context.execute_run["workload"]

        # Label each sample with the phase of the run it belongs to
        if workload == IDLE:
            for sample in self.sampler.samples:
                sample["phase"] = "idle"
        else:
            log_path = context.run_dir / "experiment.log"
            inference_start = find_log_time(log_path, INFERENCE_START_MESSAGE)
            inference_end = find_log_time(log_path, INFERENCE_END_MESSAGE)
            if inference_start is None or inference_end is None:
                raise RuntimeError(f"Could not find the inference phase in {log_path}. Did the evaluation fail?")
            for sample in self.sampler.samples:
                inside = sample["start_time"] >= inference_start and sample["end_time"] <= inference_end
                sample["phase"] = "inference" if inside else "other"

        with open(context.run_dir / "power_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.sampler.samples[0].keys()))
            writer.writeheader()
            writer.writerows(self.sampler.samples)

        output.console_log("Config.stop_measurement called!")

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""

        output.console_log("Config.stop_run() called!")

    def populate_run_data(
        self, context: RunnerContext
    ) -> Optional[Dict[str, SupportsStr]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated
        """

        phase = "idle" if context.execute_run["workload"] == IDLE else "inference"
        samples = pd.read_csv(context.run_dir / "power_samples.csv")
        samples = samples[samples["phase"] == phase]

        # During inference, the package running the inference process draws more power than the other one,
        # and the process can move between packages. Medians are used to ignore the samples taken while it moves.
        busy = samples[["cpu0_watts", "cpu1_watts"]].max(axis=1)
        other = samples[["cpu0_watts", "cpu1_watts"]].min(axis=1)

        output.console_log("Config.populate_run_data() called!")

        return {
            "phase": phase,
            "duration_seconds": float((samples["end_time"] - samples["start_time"]).sum()),
            "samples": len(samples),
            "cpu0_watts": statistics.median(samples["cpu0_watts"]),
            "cpu1_watts": statistics.median(samples["cpu1_watts"]),
            "dram0_watts": statistics.median(samples["dram0_watts"]),
            "dram1_watts": statistics.median(samples["dram1_watts"]),
            "cpu_busy_watts": statistics.median(busy),
            "cpu_other_watts": statistics.median(other),
            "package0_busy_share": float((samples["cpu0_watts"] > samples["cpu1_watts"]).mean()),
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None  # type: ignore
