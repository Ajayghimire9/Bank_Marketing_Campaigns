from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_dataset, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    frame = load_dataset(args.data)
    validate(frame)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/validated.marker").write_text(str(len(frame)), encoding="utf-8")


if __name__ == "__main__":
    main()
