"""Evaluate the thesis baseline on saved 256-dimensional HuBERT-Soft frames."""

import argparse
import json
from pathlib import Path
import pickle
import zipfile

from . import CLASS_MAP, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=CLASS_MAP)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("test_embeddings", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.model, args.checkpoint, args.test_embeddings)
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
            zipfile.BadZipFile) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
