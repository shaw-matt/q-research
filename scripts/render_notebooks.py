"""Render Quarto notebooks in parallel, then assemble the website."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_QUARTO_COMMAND = "uv run quarto"
DEFAULT_MAX_JOBS = 4


@dataclass(frozen=True)
class RenderResult:
    source: Path | None
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path.cwd() / path).resolve(strict=False)


def parse_quarto_command(raw: str) -> tuple[str, ...]:
    command = tuple(shlex.split(raw))
    if not command:
        raise ValueError("quarto command cannot be empty")
    return command


def discover_notebooks(notebook_root: Path, excludes: Sequence[Path]) -> list[Path]:
    root = normalize_path(notebook_root)
    excluded = {normalize_path(path) for path in excludes}
    if not root.exists():
        return []

    notebooks = [
        path
        for path in root.rglob("*.py")
        if path.is_file() and path.resolve(strict=False) not in excluded
    ]
    return sorted(notebooks)


def default_jobs(notebook_count: int) -> int:
    env_value = os.getenv("Q_RESEARCH_NOTEBOOK_RENDER_JOBS")
    if env_value:
        return positive_int(env_value)
    return max(1, min(notebook_count, os.cpu_count() or 1, DEFAULT_MAX_JOBS))


def run_command(command: Sequence[str], source: Path | None) -> RenderResult:
    start = time.monotonic()
    env = os.environ.copy()
    env.setdefault("QUARTO_PYTHON", sys.executable)
    try:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return RenderResult(
            source=source,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=time.monotonic() - start,
        )
    except OSError as exc:
        return RenderResult(
            source=source,
            command=tuple(command),
            returncode=127,
            stdout="",
            stderr=f"{exc}\n",
            elapsed_seconds=time.monotonic() - start,
        )


def print_result(result: RenderResult) -> None:
    target = display_path(result.source) if result.source else "project"
    status = "finished" if result.returncode == 0 else "failed"
    print(
        f"[{target}] {status} in {result.elapsed_seconds:.1f}s: "
        f"{shlex.join(result.command)}",
        flush=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def render_notebooks(
    notebooks: Sequence[Path],
    *,
    quarto_command: Sequence[str],
    jobs: int,
) -> int:
    if not notebooks:
        print("No notebooks found to render.", flush=True)
        return 0

    worker_count = max(1, min(jobs, len(notebooks)))
    print(
        f"Rendering {len(notebooks)} notebook(s) with {worker_count} parallel job(s).",
        flush=True,
    )

    failures: list[RenderResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_command,
                [*quarto_command, "render", display_path(notebook), "--no-clean"],
                notebook,
            ): notebook
            for notebook in notebooks
        }
        for future in as_completed(futures):
            result = future.result()
            print_result(result)
            if result.returncode != 0:
                failures.append(result)

    if failures:
        failed = ", ".join(display_path(result.source) for result in failures if result.source)
        print(f"Notebook render failed for: {failed}", file=sys.stderr)
        return 1
    return 0


def render_project_from_freeze(quarto_command: Sequence[str]) -> int:
    print("Rendering final Quarto project using frozen notebook results.", flush=True)
    result = run_command([*quarto_command, "render"], source=None)
    print_result(result)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-root",
        type=Path,
        default=Path("notebooks"),
        help="Directory containing Jupytext percent notebooks.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[Path("notebooks/template.py")],
        help="Notebook source path to skip. May be passed more than once.",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=None,
        help=(
            "Maximum parallel notebook renders. Defaults to "
            "Q_RESEARCH_NOTEBOOK_RENDER_JOBS or the available CPU count, capped at 4."
        ),
    )
    parser.add_argument(
        "--quarto-command",
        default=os.getenv("Q_RESEARCH_QUARTO_COMMAND", DEFAULT_QUARTO_COMMAND),
        help="Command used to invoke Quarto, parsed with shell-style quoting.",
    )
    parser.add_argument(
        "--skip-final-project-render",
        action="store_true",
        help="Skip the final project render that assembles non-notebook pages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the notebooks and commands without running Quarto.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        quarto_command = parse_quarto_command(args.quarto_command)
        notebooks = discover_notebooks(args.notebook_root, args.exclude)
        jobs = args.jobs if args.jobs is not None else default_jobs(len(notebooks))
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(f"Quarto command: {shlex.join(quarto_command)}")
        print(f"Parallel jobs: {max(1, min(jobs, len(notebooks) or 1))}")
        for notebook in notebooks:
            command = [*quarto_command, "render", display_path(notebook), "--no-clean"]
            print(f"- {shlex.join(command)}")
        if not args.skip_final_project_render:
            print(f"Final render: {shlex.join([*quarto_command, 'render'])}")
        return 0

    render_status = render_notebooks(notebooks, quarto_command=quarto_command, jobs=jobs)
    if args.skip_final_project_render or render_status != 0:
        return render_status
    project_status = render_project_from_freeze(quarto_command)
    return project_status if project_status != 0 else render_status


if __name__ == "__main__":
    raise SystemExit(main())
