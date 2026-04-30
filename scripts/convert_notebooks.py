"""Convert Jupytext percent notebooks to ipynb for render fallback."""

from pathlib import Path
import subprocess


def main() -> None:
    for path in Path("notebooks").rglob("*.py"):
        subprocess.run(
            ["uv", "run", "jupytext", "--to", "ipynb", str(path)],
            check=True,
        )


if __name__ == "__main__":
    main()
