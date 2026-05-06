# Research Notebooks

This repository contains AI-assisted research notebooks.

## Workflow

1. Create or edit notebooks in `notebooks/`.
2. Use Jupytext percent-format `.py` notebooks.
3. Render the site with Quarto.
4. Push to `main`.
5. GitHub Actions publishes the rendered site to GitHub Pages.

For this solo research workflow, small notebook changes may go directly to
`main`. The Pages workflow deploys only after the Quarto render succeeds, so a
failed render leaves the current published site unchanged until the issue is
fixed.

Before pushing notebook changes, run:

```bash
uv sync
uv run python scripts/render_notebooks.py
```

Do not push if the render fails.

## Local Setup

```bash
uv sync
uv run jupyter lab
```

## Render Site

```bash
uv run python scripts/render_notebooks.py
```

Set `Q_RESEARCH_NOTEBOOK_RENDER_JOBS` or pass `--jobs` to control the number of
parallel notebook renders.

The publish workflow first warms the Massive flat-file cache used by the
notebooks, then renders notebooks in parallel. The warm-up step prevents
multiple Quarto worker processes from downloading the same multi-year data
slices. For local cold-cache runs, use:

```bash
uv run python scripts/warm_flatfile_cache.py
uv run python scripts/render_notebooks.py
```

Set `Q_RESEARCH_FLATFILE_DOWNLOAD_WORKERS` to tune the number of concurrent S3
flat-file day downloads during cache warm-up.

## Create a New Notebook

Copy `notebooks/template.py` into a new file:

```bash
cp notebooks/template.py notebooks/examples/my-study.py
```

Then update `notebooks/index.qmd`.
