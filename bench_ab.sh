#!/bin/sh
# Compatibility entry point: use an explicit JSON case file to retain every option.
set -eu
exec python3 "$(dirname "$0")/tools/bench.py" "$@"
