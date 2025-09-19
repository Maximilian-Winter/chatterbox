# Repository Guidelines

## Project Structure & Module Organization
Chatterbox code lives in ;  and  expose the English and multilingual synthesizers, while  hosts voice conversion flows. Shared tokenizers, schedulers, and diffusers reside under . Example apps (, , , ) demonstrate CLI, batch, and Gradio usage. Benchmarks and reference plots stay beside the scripts that produced them. Tests live in the repository root and reuse bundled audio fixtures (, ) for quick smoke coverage.

## Build, Test, and Development Commands
Use  to install Chatterbox in editable mode with the pinned dependencies from . Run  for unit coverage; set  or  when you need to target specific accelerators. Execute  to exercise tokenizer and batching paths—first run may download checkpoints, so keep your cache location stable.  generates an English sample; combine  with  to validate multilingual cloning.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, snake_case names for functions and modules, and CapWords for classes. Align docstrings with the existing triple-quoted style used across . Type hints are expected on new interfaces, and optional parameters should default to production-safe values. Keep modules focused; expand within  when introducing new model families or preprocessors.

## Testing Guidelines
Keep fast logic in -discoverable functions named . Mock heavyweight components by stubbing  or supplying small tensors; reserve real checkpoint loads for opt-in scripts such as . Capture temporary waveforms under a throwaway directory (e.g., ) and clean them after assertions to avoid polluting the repo.

## Commit & Pull Request Guidelines
Write imperative commit subjects like  and aim for ≤50 characters. In pull requests, link related issues, describe user-facing impact, and attach audio diffs or benchmark deltas whenever synthesis quality changes. Document validation steps—commands run, hardware targets, environment variables—so reviewers can reproduce results quickly.

## Environment & Asset Management
Keep large model weights out of version control; rely on the built-in download utilities and document expected cache directories. Safeguard API keys or premium endpoints by loading them from environment variables rather than hardcoding. When sharing demo outputs, compress waveforms and note the originating commit for traceability.
