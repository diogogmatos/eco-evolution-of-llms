import os
import random
import subprocess
import sys
import time
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

random.seed(12345)  # For reproducibility of shuffling, if enabled in RunTableModel

from utils.models import download_model, update_benchmark_config


class RunnerConfig:
    CURRENT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name: str = "test_run"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path: Path = ROOT_DIR / "results"

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type: OperationType = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms: int = 0

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

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        models = pd.read_csv(ROOT_DIR / "model-selection/selected_models.csv")
        model = FactorModel("model", models["model"].unique().tolist())
        benchmark = FactorModel("benchmark", ["MMLU-Pro"])
        self.run_table_model = RunTableModel(
            factors=[model, benchmark],
            exclude_combinations=[],
            data_columns=["runtime", "accuracy"],
            repetitions=1,
            shuffle=True,
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""

        self.env = os.environ.copy()
        self.models = pd.read_csv(ROOT_DIR / "model-selection/selected_models.csv")

        # Ensure models are downloaded and quantized before starting the experiment.
        for _, model in self.models.iterrows():
            download_model(model["model"].split(" ")[0])

        # Ensure config file is updated with the creator-provided settings.
        update_benchmark_config(
            model_list=self.models["model"].unique().tolist(),
            config_path=ROOT_DIR / "experiments/model_configs.json",
        )

        output.console_log("Config.before_experiment() called!")

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""

        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""

        output.console_log("Config.start_run() called!")

        model = context.execute_run["model"]
        benchmark = context.execute_run["benchmark"]
        max_model_length = int(
            self.models[self.models["model"] == model].iloc[0]["max_model_length"]
        )
        is_thinking = bool(
            self.models[self.models["model"] == model].iloc[0]["is_thinking"]
        )
        supports_enable_thinking = bool(
            self.models[self.models["model"] == model].iloc[0][
                "supports_enable_thinking"
            ]
        )

        cmd = ""
        if benchmark == "MMLU-Pro":
            eval_script = ROOT_DIR / "dependencies/MMLU-Pro/evaluate_from_local.py"
            cmd = f"python {eval_script} --model {model.split(' ')[0]} --selected_subjects history --save_dir {context.run_dir} --global_record_file {Path(dirname(context.run_dir)) / 'eval_record_collection.csv'} --config_path {ROOT_DIR / 'experiments/model_configs.json'} --max_model_length {min(max_model_length, 10240) if max_model_length != "N/A" else ''} --max_thinking_tokens {4096 if is_thinking else 0} {f'--enable_thinking {is_thinking}' if supports_enable_thinking else ''} --ntrain {0 if is_thinking else 5} --sample"
        else:
            raise NotImplementedError(f"Benchmark {benchmark} not implemented yet!")

        log = open(f"{context.run_dir}/experiment.log", "w")

        output.console_log(f"Running command: {cmd}")

        start = time.time()
        subprocess.run(cmd, shell=True, stdout=log, stderr=log, env=self.env)
        end = time.time()

        runtime = end - start
        with open(context.run_dir / "runtime.txt", "w") as f:
            f.write(str(runtime))

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""

        output.console_log("Config.start_measurement() called!")

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""

        output.console_log("Config.interact() called!")

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""

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

        output.console_log("Config.populate_run_data() called!")

        benchmark = context.execute_run["benchmark"]

        run_data: Dict[str, SupportsStr] = {}
        with open(context.run_dir / "runtime.txt", "r") as f:
            runtime = float(f.read().strip())
            run_data["runtime"] = runtime

        if benchmark == "MMLU-Pro":
            summary_file = next((context.run_dir / "summary").glob("*.txt"))

            with open(summary_file, "r") as f:
                content = f.read()
                accuracy = float(
                    content.strip().split("\n")[-1].strip().split(": ")[-1]
                )
                run_data["accuracy"] = accuracy
        else:
            raise NotImplementedError(f"Benchmark {benchmark} not implemented yet!")

        return run_data

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None  # type: ignore
