# Research Notebooks

This repository contains AI-assisted research notebooks.

## Workflow

1. Create or edit notebooks in `notebooks/`.
2. Use Jupytext percent-format `.py` notebooks.
3. Render the site with Quarto.
4. Push to `main`.
5. GitHub Actions publishes the rendered site to GitHub Pages.

## Local Setup

```bash
uv sync
uv run jupyter lab
```

## Render Site

```bash
QUARTO_PYTHON=.venv/bin/python quarto render
```

## Create a New Notebook

Copy `notebooks/template.py` into a new file:

```bash
cp notebooks/template.py notebooks/examples/my-study.py
```

Then update `notebooks/index.qmd`.
