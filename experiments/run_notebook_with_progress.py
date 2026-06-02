# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run a Jupyter notebook cell-by-cell as plain Python with timing markers.

This script exists because debugging a hanging notebook in CI is awful:
``jupyter nbconvert --execute`` prints ``[NbConvertApp] Converting notebook
... to notebook`` and then NOTHING for minutes while the kernel runs cells
silently. When the runner cancels the step at some opaque threshold (e.g.,
GitHub-hosted ubuntu-latest's "Error: The operation was canceled."),
there's no signal pointing at the offending cell.

This helper bypasses Jupyter entirely: it parses the .ipynb, runs each
code cell sequentially in a single Python process with ``python -u``-style
unbuffered output, and prints a marker BEFORE and timing AFTER each cell.
Whichever cell hangs surfaces as the last ``=== cell N: ... ===`` line
printed before the runner cancels.

Usage (local-only — no longer wired into CI; the ``Notebooks`` workflow
now runs only ``nbconvert --execute``. Re-add the workflow step or run
this helper locally if a notebook ever hangs in CI again.):
    python experiments/run_notebook_with_progress.py path/to/notebook.ipynb

    # With a per-cell wall-clock budget (default: 600s, same as
    # nbconvert's --ExecutePreprocessor.timeout). Cells exceeding the
    # budget cause this script to exit non-zero with a clear error
    # printing the cell's source.
    python experiments/run_notebook_with_progress.py \\
        path/to/notebook.ipynb --per-cell-timeout 60

This script is also wired into the .github/workflows/notebooks.yml
mode-dependent matrix as the FIRST notebook-execution step. If it fails,
it fails fast with cell-level info; if it succeeds, the subsequent
nbconvert step is highly likely to succeed too (it executes the same
cells, just through a Jupyter kernel instead of plain exec).

What this script does NOT do:
    * Render outputs back into the .ipynb. nbconvert with --inplace does
      that (and is what the workflow runs after this script passes).
    * Execute IPython magics (``%matplotlib``, ``%timeit``, etc.) -
      those cells are skipped with a printed note.
    * Validate kernel-launch behavior. If you suspect the Jupyter kernel
      is broken, run a trivial nbconvert separately.
"""

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path


def _cell_preview(src: str, max_chars: int = 80) -> str:
    """Return the first non-blank line of the cell, truncated."""
    for line in src.splitlines():
        line = line.strip()
        if line:
            return line[:max_chars]
    return ""


def _alarm_handler(signum, frame):
    raise TimeoutError("per-cell timeout")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebook", type=Path)
    p.add_argument(
        "--per-cell-timeout",
        type=int,
        default=600,
        help=(
            "Wall-clock budget per cell in seconds (default: 600, same as "
            "nbconvert's --ExecutePreprocessor.timeout). Cells exceeding "
            "this budget cause a non-zero exit with the offending cell's "
            "source printed."
        ),
    )
    args = p.parse_args()

    if not args.notebook.exists():
        print(f"::error::notebook not found: {args.notebook}", flush=True)
        return 2

    nb = json.loads(args.notebook.read_text())
    ns = {"__name__": "__main__"}

    # SIGALRM-based per-cell timeout. Only on Unix; on Windows the timeout
    # is silently disabled (this script is primarily for Linux/Mac CI).
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)

    print(
        f"running {args.notebook} cell-by-cell with per-cell timeout "
        f"{args.per_cell_timeout}s",
        flush=True,
    )

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if src.lstrip().startswith("%"):
            print(
                f"=== cell {i}: skipping IPython magic "
                f"({_cell_preview(src)!r}) ===",
                flush=True,
            )
            continue

        print(f"=== cell {i}: {_cell_preview(src)!r} ===", flush=True)
        t0 = time.perf_counter()
        if has_alarm:
            signal.alarm(args.per_cell_timeout)
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except TimeoutError:
            elapsed = time.perf_counter() - t0
            print(
                f"\n::error::cell {i} exceeded per-cell timeout "
                f"({args.per_cell_timeout}s); ran for {elapsed:.1f}s.",
                flush=True,
            )
            print("--- offending cell source ---", flush=True)
            print(src, flush=True)
            return 1
        except SystemExit:
            raise
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            print(
                f"\n::error::cell {i} raised {type(e).__name__} after "
                f"{elapsed:.2f}s: {e}",
                flush=True,
            )
            print("--- traceback ---", flush=True)
            traceback.print_exc()
            print("--- offending cell source ---", flush=True)
            print(src, flush=True)
            return 1
        finally:
            if has_alarm:
                signal.alarm(0)
        print(
            f"    cell {i} done in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    print("=== ALL CELLS DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
