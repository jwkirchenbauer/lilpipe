#!/usr/bin/env bash
set -eu
date -u
cat /proc/2028821/io
timeout 30 gdb -nx -q -batch -ex 'set pagination off' -ex 'set debuginfod enabled off' -ex 'set auto-solib-add off' -ex 'attach 2028821' -ex 'sharedlibrary libamdhip64' -ex 'thread 1' -ex 'bt 18' -ex 'detach'
