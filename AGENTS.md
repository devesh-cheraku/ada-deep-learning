# Repository Guidelines

## Project Structure & Module Organization
`homework1_jun_6/` contains all deliverables. Core implementation in `homework/` with modules `bignet.py`, `half_precision.py`, `lora.py`, `low_precision.py`, `qlora.py`, `lower_precision.py`; each exposes specialized layers or `load` helpers. `grader/` holds the lightweight harness (`grader.py`, `tests.py`) executed via `python3 -m grader`. `bundle.py` packages submissions, while `bignet.pth` supplies the reference weights. Keep any experimental notebooks or scratch scripts outside this directory.

## Environment, Build & Test Commands
Create the course environment with:
```
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
pip install -r requirements.txt
```
Routine commands:
- `python3 -m homework.stats bignet half_precision` (memory comparison table)
- `python3 -m homework.compare bignet lora` (numerical diff check)
- `python3 -m homework.fit lora` (quick overfit sanity run)
- `python3 -m grader homework -v` (local subset of grading rubric)
- `python3 bundle.py homework <UTID>` (submission archive)

## Coding Style & Naming Conventions
Stick to PEP 8 with 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, classes in `CapWords`, functions and files in `snake_case`. Annotate tensor shapes with type hints where feasible (`def forward(self, x: torch.Tensor) -> torch.Tensor`). Prefer explicit helper modules to ad-hoc scripts, and avoid adding dependencies beyond PyTorch and the packages in `requirements.txt`.

## Testing Guidelines
Before opening a pull request, confirm each variant matches BigNet outputs within tolerance using `homework.compare`, and capture memory deltas with `homework.stats`. Extend `grader/tests.py` if you add behaviors that need coverage—new checks should instantiate a `Case` mirroring the existing patterns. Record the command outputs so reviewers can reproduce the results quickly.

## Commit & Pull Request Expectations
Write concise, imperative commit subjects (e.g., `Add 4-bit quantized linear layer`), followed by an optional body outlining design decisions. Pull requests should describe the intent, list validation commands run, and attach abbreviated logs for any new metrics or training loops. Reference related Canvas discussions or issue IDs when available, and flag any breaking API changes up front.

## Configuration & Assets
Do not modify `bignet.pth`; treat it as read-only reference data. When experimenting with larger checkpoints, keep them untracked or store paths in `.gitignore`. Document any environment variables or hardware assumptions inside your PR description so graders can rerun the experiments reliably.
