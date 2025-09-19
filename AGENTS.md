# Repository Guidelines

## Project Structure & Module Organization
- Core Python packages live under ; , , and  expose the primary inference APIs while  contains model-specific utilities.
- Top-level helper scripts (, , ) demonstrate CLI and Gradio usage; benchmark scripts and  outputs document performance studies.
- Test entry points (, , ) sit in the repository root alongside sample audio assets (, ) used for smoke checks.

## Build, Test, and Development Commands
-  installs the package in editable mode with all runtime deps from .
-  runs the lightweight unit checks; use  or  to target hardware if needed.
-  executes the heavier integration suite that exercises tokenizer, batching, and voice cloning flows; expect larger model downloads on first run.
-  is a quick sanity check for English TTS; adjust  when using  for multilingual voices.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents, descriptive snake_case for functions/modules, and CapWords for classes; mirror existing docstring style when adding public APIs.
- Type hints are expected on new interfaces in , and optional arguments should default to sensible production values.
- Keep modules small; introduce subpackages inside  when adding model families or tokenizers.

## Testing Guidelines
- Prefer  function-style tests under the repository root; name files  so discovery remains automatic.
- Mock heavyweight model calls by stubbing  or using small tensors; integration scripts may rely on real checkpoints but gate them behind environment checks.
- Aim for covering critical audio paths (tokenization, batch scheduling, waveform synthesis) and document any skipped tests with TODOs.

## Commit & Pull Request Guidelines
- Write imperative, informative commit subjects (e.g., ) that summarize the behavior change in 50 characters or less.
- Link PRs to issues when available, describe user-facing impact, and attach audio diffs or benchmark summaries when behavior shifts.
- Include validation notes (commands run, hardware used) in PR descriptions so reviewers can reproduce results quickly.
