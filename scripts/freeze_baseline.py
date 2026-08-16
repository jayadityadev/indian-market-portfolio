"""Freeze and verify current dataset/model lineage for reproducible experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_registry import build_registry, verify_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/model_registry.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        registry = json.loads(args.output.read_text(encoding="utf-8"))
        errors = verify_registry(registry)
    else:
        registry = build_registry(args.data_dir, args.models_dir)
        args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        errors = verify_registry(registry)

    print(json.dumps({"valid": not errors, "errors": errors, "registry": registry}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
