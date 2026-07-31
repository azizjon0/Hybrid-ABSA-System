"""Command-line entry point for the Hybrid ABSA pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_LLM_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_THRESHOLD,
)
from llm import create_openai_client
from pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hybrid ABSA pipeline")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing a text column",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-module",
        default="model",
        help="Python module exposing a variable named model",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_openai_client(args.disable_llm)

    result = run_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_module=args.model_module,
        threshold=args.threshold,
        llm_model=args.llm_model,
        checkpoint_every=args.checkpoint_every,
        client=client,
    )

    print("\nFinished")
    print(f"Output rows: {len(result)}")
    corrected = int(result["llm_called"].sum()) if not result.empty else 0
    print(f"LLM-corrected rows: {corrected}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
