"""Render the compiled coordinator graph's structure.

Usage:
    python -m src.graph.visualize              # print Mermaid syntax
    python -m src.graph.visualize --png out.png  # also save a PNG
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.graph.build import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", metavar="PATH", help="Also save a PNG to this path.")
    args = parser.parse_args()

    graph = build_graph()
    drawable = graph.get_graph()

    print(drawable.draw_mermaid())

    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = drawable.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"\nSaved PNG to {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
