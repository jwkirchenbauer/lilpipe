#!/usr/bin/env bash
set -eu
date -u
awk '/^(State|VmRSS|VmSize|Threads):/' /proc/2028821/status
cat /proc/2028821/io
timeout 30 gdb -q -batch -p 2028821 -ex 'set pagination off' -ex 'thread 1' -ex 'bt 24' -ex 'detach'
