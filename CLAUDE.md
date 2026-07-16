# CLAUDE.md — mothra-text / line-seg-eval

## Python environment

All pipeline code (`run_pipeline.py`, `run_kraken.py`, etc.) must be run with the
`line-seg-eval` conda environment. **Do not use `conda run -n line-seg-eval python`** —
pyenv intercepts the `python` command and uses `/Users/cassiebastress/.pyenv/versions/3.12.6/`
instead of the conda env, which is missing `htrflow` and other dependencies.

Always invoke Python directly via the full path:
```
/Users/cassiebastress/miniconda3/envs/line-seg-eval/bin/python
```

The pyenv Python 3.12 has `kraken` installed but lacks `htrflow`, `biopython`, and
`volpiano-display-utilities`. The conda env Python 3.10 has everything.

## Dependency manifest

All dependencies are pinned in `requirements.txt` (generated from `pip freeze` in the
conda env). Install with `pip install -r requirements.txt` inside the `line-seg-eval`
conda env. To update after adding a new package, re-run `pip freeze > requirements.txt`
and commit the result.

`pyproject.toml` contains project metadata and pytest configuration only — it does not
manage dependencies and does not require Poetry.

## Documentation rule

Before committing any code change, update the relevant documentation:
- If the change affects how a step works, update `steps/README.md`.
- If the change affects CLI flags or pipeline behaviour visible to users, update the root `README.md`.
- If the change affects how the GUI JSON is generated or interpreted, update `gui/README.md`.
- If the change affects a script in `scripts/`, update `scripts/README.md`.
- If the change adds or removes a dependency, re-run `pip freeze > requirements.txt` in the conda env and commit the updated file.

Docs must be updated in the same commit as the code change, not as a follow-up.
