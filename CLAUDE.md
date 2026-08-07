# mothra-text

## Python environment

Never use `conda run -n line-seg-eval python` — pyenv intercepts `python` and uses the wrong interpreter. Always use the full path:

```
/Users/cassiebastress/miniconda3/envs/line-seg-eval/bin/python
```

The pyenv Python 3.12 has `kraken` but lacks `htrflow`, `biopython`, and `volpiano-display-utilities`. The conda env Python 3.10 has everything.

## Tests

```bash
/Users/cassiebastress/miniconda3/envs/line-seg-eval/bin/python -m pytest tests/ -v
```

Run after any change to `steps/` or `run_pipeline.py`. Run a single test file when faster iteration is needed.

## Key commands

```bash
# Single-folio pipeline
/Users/cassiebastress/miniconda3/envs/line-seg-eval/bin/python run_pipeline.py \
  --image IMAGE --folio FOLIO --source-id SOURCE_ID --output-dir DIR

# Multi-folio chain
/Users/cassiebastress/miniconda3/envs/line-seg-eval/bin/python run_chain.py \
  --images ... --folios ... --output-dir DIR
```

## Output rules

- MEI JSON auto-naming: `{RISM-code}_{shelfmark}_{folio}.json` (e.g. `CH-E_611_001r.json`) — always zero-pad folios (`001r` not `1r`)
- `--export-json` (Pipeline Inspector JSON): save to `~/Downloads/DDMAL/`, never to `gui/public/`
- Never overwrite an existing output file — always use a distinct path unless explicitly told to overwrite

## Dependencies

All dependencies are pinned in `requirements.txt` (from `pip freeze` inside the conda env). `pyproject.toml` is metadata and pytest config only — not managed by Poetry.

When adding or removing a package:
```bash
pip freeze > requirements.txt  # run inside conda env
```
Commit the updated `requirements.txt` with the code change.

## Documentation rule

Update the relevant doc in the same commit as the code change — never as a follow-up:

| What changed | Update |
|---|---|
| A pipeline step | `steps/README.md` |
| CLI flags or pipeline behaviour | root `README.md` |
| GUI JSON schema | `gui/README.md` |
| A script in `scripts/` | `scripts/README.md` |
| A dependency | re-run `pip freeze > requirements.txt` and commit |
