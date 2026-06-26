# AGENTS.md

## Project Overview
This project is evolving implementation of PySINDy and PySR to see if they can derrive a stable equations for LFP data modality.

## Goals
- Production pipeline belongs in `scripts/`.

## Coding Style
- Use Python 3.12+
- Add short comments for each section of the document
- Google-style docstrings

## Architecture
- Data loading in load_data/
- SINDy and PySR models in models/
- Visualization in outputs/

## Rules
- Never duplicate code.
- Never hardcode file paths.
- Every public function needs a docstring.
- Never change analysis assumptions without explicitly documenting them
- Use dataclasses when appropriate
- Do not edit README.md without explicit instruction to do so.

## Scientific Assumptions
- Sampling frequency = 500 Hz.
- Delayed embeddings should use sklearn TimeDelayEmbedding.
- No filtering inside plotting functions

When making significant changes:

1. Explain the reasoning.
2. Preserve backwards compatibility.
3. Prefer minimal diffs.
4. Ask before changing scientific assumptions.
5. Never silently remove code.