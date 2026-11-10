#!/usr/bin/env bash
set -eu
date -u
ps -u kirchenb -o pid,ppid,stat,etime,pcpu,pmem,comm,wchan:28
/opt/rocm-7.2.0/bin/rocm-smi --showuse --showmemuse --showmeminfo vram --showpids
