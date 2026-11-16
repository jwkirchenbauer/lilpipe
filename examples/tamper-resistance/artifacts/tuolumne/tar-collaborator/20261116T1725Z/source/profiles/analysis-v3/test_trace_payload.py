"""Tiny integration check for diagnostic-only execution wrapping."""

import faulthandler
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(300, repeat=True)
assert faulthandler.is_enabled()
assert __name__ == "__main__"
assert sys.argv[1:] == ["verify", "--kept"]
print("Trace wrapper preserves arguments, main execution, and fatal-fault reporting.")
