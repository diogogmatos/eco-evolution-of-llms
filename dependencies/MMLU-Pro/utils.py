import os
import subprocess
import psutil
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time


def get_gpu_temp():
    """Queries nvidia-smi for the current temperature of the L40S GPU."""
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return int(output)
    except Exception as e:
        print(f"Error reading GPU temp: {e}")
        return 0


def get_cpu_temp():
    """Reads the CPU core/package temperature from the Linux sysfs interface."""
    try:
        # psutil maps directly to /sys/class/thermal/ or lm-sensors
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            # Returns the Package id 0 or highest core temperature
            return int(temps["coretemp"][0].current)
        elif "cpu_thermal" in temps:
            return int(temps["cpu_thermal"][0].current)
        else:
            # Fallback if psutil naming differs on your specific kernel
            for name, entries in temps.items():
                if entries:
                    return int(entries[0].current)
    except Exception as e:
        print(f"Error reading CPU temp via psutil: {e}")
    return 0


def _burn_core(burn_duration=5):
    """An infinite math loop for a single core (run for a short burst)."""
    # Simple, high-intensity arithmetic loop
    end_time = time.time() + burn_duration
    while time.time() < end_time:
        _ = 123456.78 / 987.65


def cpu_burn_step(burn_duration=5):
    """Spawns a process for every hardware thread to maximize CPU Package heat instantly."""
    cores = cpu_count()
    with Pool(cores) as p:
        p.map(_burn_core, [burn_duration] * cores)


def async_cpu_burn_step(burn_duration=5):
    """Asynchronously spawns a process for every hardware thread to maximize CPU Package heat instantly."""
    cores = cpu_count()
    p = Pool(cores)
    p.map_async(_burn_core, [burn_duration] * cores)
    p.close()


def cpu_heavy_worker():
    """
    Infinite loop running massive CPU-bound matrix dot products.
    Forces continuous AVX-512 / AMX instruction execution on the Xeon cores.
    """
    # Prevent NumPy from trying to multi-thread internally within a single process,
    # which can cause thread thrashing. We handle parallelization explicitly.
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # 4000x4000 fills the CPU L3 cache and forces heavy silicon execution
    size = 4000
    while True:
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        _ = np.dot(a, b)
