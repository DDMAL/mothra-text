"""Run all three line segmentation models on folio images."""

import argparse
import os
import subprocess
import sys

PYTHON = sys.executable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DEFAULT_FOLIOS = os.path.join(_ROOT, "data", "folios")
_DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folios", default=_DEFAULT_FOLIOS)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    scripts = [
        os.path.join(_HERE, "run_htrflow.py"),
        os.path.join(_ROOT, "run_kraken.py"),
    ]
    for script in scripts:
        result = subprocess.run([
            PYTHON, script,
            "--folios", args.folios,
            "--output", args.output,
        ])
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
