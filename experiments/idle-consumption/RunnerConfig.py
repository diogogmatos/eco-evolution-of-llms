import glob
import os
import sys
import pynvml
import time
import json

from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

RAPL_DIR = "/sys/class/powercap"


def find_rapl_domains():
    # The server has two CPU packages, each with its own DRAM sub-domain. All of them are measured.
    cpu_paths, dram_paths = [], []
    for domain in sorted(glob.glob(os.path.join(RAPL_DIR, "intel-rapl:*"))):
        with open(os.path.join(domain, "name"), "r") as f:
            name = f.read().strip()
        if name.startswith("package"):
            cpu_paths.append(os.path.join(domain, "energy_uj"))
        elif name == "dram":
            dram_paths.append(os.path.join(domain, "energy_uj"))
    return cpu_paths, dram_paths


def read_rapl_counters(paths):
    readings = []
    for path in paths:
        with open(path, "r") as f:
            readings.append(int(f.read().strip()))
    return readings


cpu_rapl_paths, dram_rapl_paths = find_rapl_domains()
# Without this check, unreadable counters would silently record zero CPU and DRAM energy
if not cpu_rapl_paths or not dram_rapl_paths:
    raise RuntimeError(f"No RAPL package or DRAM counters found in {RAPL_DIR}.")
gpu_handle = None


def read_rapl_max_range(energy_path):
    with open(energy_path.replace("energy_uj", "max_energy_range_uj"), "r") as f:
        return int(f.read().strip())


def rapl_deltas(start_uj, end_uj, max_ranges_uj):
    # Energy of each of the given domains (one reading per CPU package)
    deltas = []
    for start, end, max_range in zip(start_uj, end_uj, max_ranges_uj):
        delta = end - start
        # The counter restarted from zero during the interval. A single interval never
        # lasts long enough for the counter to wrap more than once (~50 min at 90 W).
        if delta < 0:
            delta += max_range
        deltas.append(delta)
    return deltas


CPU_RAPL_MAX_RANGES = [read_rapl_max_range(path) for path in cpu_rapl_paths]
DRAM_RAPL_MAX_RANGES = [read_rapl_max_range(path) for path in dram_rapl_paths]


def get_status():
    global gpu_handle

    # Check if NVML needs initialization in this specific process context
    if gpu_handle is None:
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    cpu_uj = read_rapl_counters(cpu_rapl_paths)
    dram_uj = read_rapl_counters(dram_rapl_paths)

    gpu_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)

    return cpu_uj, dram_uj, gpu_mj


ROOT_DIR = Path(dirname(dirname(dirname(realpath(__file__)))))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.others import get_gpu_temp, get_cpu_temp


class RunnerConfig:
    CURRENT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name: str = "idle_consumption"

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
        self.run_table_model = RunTableModel(
            factors=[],
            exclude_combinations=[],
            data_columns=[
                "runtime_seconds",
                "total_cpu_joules",
                "total_dram_joules",
                "total_gpu_joules",
                "avg_cpu_power_watts",
                "avg_dram_power_watts",
                "avg_gpu_power_watts",
                "gpu_temperature_celsius",
                "cpu_temperature_celsius",
                # Power of each CPU package and of its DRAM, to allow comparisons with a single package
                *[f"avg_cpu_{i}_power_watts" for i in range(len(cpu_rapl_paths))],
                *[f"avg_dram_{i}_power_watts" for i in range(len(dram_rapl_paths))],
            ],
            repetitions=10,
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.before_experiment() called!")

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""

        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""

        start = time.time()
        cpu_start, dram_start, gpu_start = get_status()
        time.sleep(30)  # Idle for 30 seconds
        gpu_temp, cpu_temp = get_gpu_temp(), get_cpu_temp()
        cpu_end, dram_end, gpu_end = get_status()
        end = time.time()

        with open(context.run_dir / "data.json", "w") as f:
            runtime_seconds = end - start
            # Convert from microjoules to joules
            cpu_joules = [
                delta / 1e6
                for delta in rapl_deltas(cpu_start, cpu_end, CPU_RAPL_MAX_RANGES)
            ]
            dram_joules = [
                delta / 1e6
                for delta in rapl_deltas(dram_start, dram_end, DRAM_RAPL_MAX_RANGES)
            ]
            total_cpu_joules = sum(cpu_joules)
            total_dram_joules = sum(dram_joules)
            total_gpu_joules = (
                gpu_end - gpu_start
            ) / 1000.0  # Convert from millijoules to joules
            data = {
                "runtime_seconds": runtime_seconds,
                "total_cpu_joules": total_cpu_joules,
                "total_dram_joules": total_dram_joules,
                "total_gpu_joules": total_gpu_joules,
                "avg_cpu_power_watts": total_cpu_joules / runtime_seconds,
                "avg_dram_power_watts": total_dram_joules / runtime_seconds,
                "avg_gpu_power_watts": total_gpu_joules / runtime_seconds,
                "gpu_temperature_celsius": gpu_temp,
                "cpu_temperature_celsius": cpu_temp,
                **{
                    f"avg_cpu_{i}_power_watts": joules / runtime_seconds
                    for i, joules in enumerate(cpu_joules)
                },
                **{
                    f"avg_dram_{i}_power_watts": joules / runtime_seconds
                    for i, joules in enumerate(dram_joules)
                },
            }
            json.dump(data, f, indent=4)

        output.console_log("Config.start_run() called!")

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

        run_data: Dict[str, SupportsStr] = {}
        with open(context.run_dir / "data.json", "r") as f:
            data = json.load(f)
            run_data = {
                **data,
            }

        output.console_log("Config.populate_run_data() called!")
        return run_data

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None  # type: ignore
