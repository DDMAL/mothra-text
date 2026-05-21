"""Run all three stages of the PyLaia baseline experiment in sequence."""

import os
import subprocess
import sys

PYTHON = sys.executable
_HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    os.path.join(_HERE, "01_segment.py"),
    os.path.join(_HERE, "02_extract_crops.py"),
    os.path.join(_HERE, "03_run_pylaia.py"),
]


def main():
    for script in STAGES:
        print(f"\n{'='*60}")
        print(f"Running {os.path.basename(script)}")
        print(f"{'='*60}")
        result = subprocess.run([PYTHON, script])
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
