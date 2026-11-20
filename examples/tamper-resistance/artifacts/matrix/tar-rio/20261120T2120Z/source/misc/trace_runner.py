"""Run the frozen numerical entry point without periodic asynchronous stack walks."""

import faulthandler
import runpy
import sys


def suppress_periodic_dump(*args, **kwargs):
    print(
        "Periodic asynchronous stack dumping disabled for this retry; fatal-fault reporting remains enabled.",
        flush=True,
    )


faulthandler.dump_traceback_later = suppress_periodic_dump
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
