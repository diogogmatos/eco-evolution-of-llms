import psutil
import subprocess


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
            for _, entries in temps.items():
                if entries:
                    return int(entries[0].current)
    except Exception as e:
        print(f"Error reading CPU temp via psutil: {e}")
    return 0
