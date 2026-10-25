"""Record generated VNNI encodings and possible stack references for manual review."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
run = subprocess.run(
    ["objdump", "-d", "-M", "intel", str(args.binary)],
    capture_output=True,
    text=True,
    check=True,
)
functions = {}
current = "unknown"
for line in run.stdout.splitlines():
    match = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
    if match:
        current = match[1]
    if re.match(r"(vnni|sign|nibble|pair_micro)", current):
        record = functions.setdefault(
            current,
            {
                "vpdpbusd": 0,
                "vex_vpdpbusd": 0,
                "evex_vpdpbusd": 0,
                "stack_reference_lines": 0,
            },
        )
        if "vpdpbusd" in line:
            record["vpdpbusd"] += 1
            raw = line.split(":", 1)[1].lstrip()
            if raw.startswith("c4 "):
                record["vex_vpdpbusd"] += 1
            if raw.startswith("62 "):
                record["evex_vpdpbusd"] += 1
        if "[rsp" in line or "[rbp" in line:
            record["stack_reference_lines"] += 1
args.output.write_text(
    json.dumps(
        {
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "functions": functions,
            "note": "Stack references are diagnostics, not a proof of register spilling or speed.",
        },
        indent=2,
    )
    + "\n"
)
print(args.output)
