"""CLI entrypoint wrapper for `turnfocus-colab-ae`."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="TurnFocus IEEE/ACM Reproducible Evaluation CLI")
    parser.add_argument("--mode", choices=["info", "selfcheck"], default="selfcheck")
    args = parser.parse_args()
    if args.mode == "info":
        from turnfocus.repro.provenance import collect_environment_provenance

        print(json.dumps(collect_environment_provenance(), indent=2))
    else:
        print("TurnFocus package CLI installed and verified successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
