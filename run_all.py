"""Run all three line segmentation models on folio images."""

import subprocess
import sys

PYTHON = sys.executable


def main():
    for script in ("run_htrflow.py", "run_kraken.py"):
        result = subprocess.run([PYTHON, script])
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
