from __future__ import annotations

from pathlib import Path
import os
import pandas as pd

from Plugins.Profilers.DataSource import CLISource, ParameterDict, ValueRef

JOULARCORE_PARAMETERS = {
    ("-p", "--pid"): int,
    ("-a", "--app"): str,
    ("-f", "--file"): ValueRef,
    ("-c", "--component"): str,   # cpu | gpu | none (default: none = cpu+gpu)
    ("-r", "--ringbuffer"): None,
    ("-h", "--help"): None,
    ("-V", "--version"): None,
}


class JoularCore(CLISource):
    """
    Joular Core integration for experiment-runner.

    Supported operational modes:
      1) app flag only -> monitor an app by name
      2) pid flag only -> monitor a process by PID
      3) no pid/app flag -> monitor whole system (default)

    Notes:
      - If both -a and -p are provided, Joular Core will emit an error itself.
    """

    parameters = ParameterDict(JOULARCORE_PARAMETERS)
    source_name = "joularcore"
    supported_platforms = ["Linux", "Darwin", "Windows"]

    def __init__(
        self,
        out_file: Path = Path("joularcore.csv"),
        app: str | None = None,
        pid: int | None = None,
        component: str | None = None,        # "cpu" | "gpu" | "none"
        ringbuffer: bool = False,
        additional_args: dict = {},
    ):
        super().__init__()

        self.requires_admin = True
        self.logfile = Path(out_file) if out_file else None

        # Base args
        self.args = {}
        if self.logfile:
            self.args.update({"-f": self._logfile})
        if ringbuffer:
            self.update_parameters(add={"-r": None})
        if component:
            self.update_parameters(add={"-c": component})

        # Apply explicit convenience selectors (if provided)
        if app is not None:
            self.update_parameters(add={"-a": app})
        if pid is not None:
            self.update_parameters(add={"-p": int(pid)})

        # Allow caller overrides/extensions (including -a/-p if desired)
        self.update_parameters(add=additional_args)

    @staticmethod
    def parse_log(logfile: Path):
        # Joular Core already outputs CSV. Strip spaces after delimiter for stable column naming.
        return pd.read_csv(logfile, skipinitialspace=True).to_dict()
