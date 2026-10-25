"""Regenerate the four-digit read-only table deterministically."""

from pathlib import Path

text = "".join(f"{i:04d}" for i in range(10000))
path = Path(__file__).resolve().parents[1] / "lab/quad_table.h"
path.write_text(
    "/* Generated from decimal 0000..9999 by tools/generate_tables.py. */\nstatic const char static_quad[40001] =\n"
    + "".join('"' + text[i : i + 100] + '"\n' for i in range(0, len(text), 100))
    + ";\n"
)
