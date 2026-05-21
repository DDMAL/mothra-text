"""Run all three line segmentation models on folio images."""

import argparse
import os
import subprocess
import sys

PYTHON = sys.executable
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FOLIOS = os.path.join(_HERE, "data", "folios")
_DEFAULT_OUTPUT = os.path.join(_HERE, "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folios", default=_DEFAULT_FOLIOS)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    for script in ("run_htrflow.py", "run_kraken.py"):
        result = subprocess.run([
            PYTHON, os.path.join(_HERE, script),
            "--folios", args.folios,
            "--output", args.output,
        ])
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
