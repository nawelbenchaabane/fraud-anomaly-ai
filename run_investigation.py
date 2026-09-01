from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline import InvestigationPipeline
from src.reporting import (
    generate_llm_report,
    local_report_from_packet,
    report_to_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score a BankSim transaction and build an "
            "analyst-facing investigation report."
        )
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--test-index",
        type=int,
        help=(
            "Positional index in the temporal test split "
            "(step >= 150)."
        ),
    )

    source.add_argument(
        "--json",
        type=Path,
        help=(
            "Path to a JSON object containing one raw transaction."
        ),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory.",
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="Number of Autoencoder feature errors to include.",
    )

    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Generate the final report through the OpenAI API. "
            "Otherwise use the deterministic local renderer."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional OpenAI model override. "
            "Otherwise OPENAI_MODEL or gpt-5.6-luna is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cli"),
        help="Directory for packet/report files.",
    )

    return parser.parse_args()


def load_json_transaction(
    path: Path,
) -> pd.DataFrame:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(
            f
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "--json must contain one JSON object."
        )

    return pd.DataFrame(
        [payload]
    )


def main():
    args = parse_args()

    project_root = args.project_root.resolve()

    pipeline = InvestigationPipeline(
        project_root=project_root,
        device="cpu",
    )

    if args.test_index is not None:
        packet = pipeline.analyze_test_position(
            test_position=args.test_index,
            top_ae_features=args.top_features,
        )
    else:
        transaction = load_json_transaction(
            args.json
        )

        packet = pipeline.analyze_dataframe(
            transaction,
            transaction_ids=[0],
            top_ae_features=args.top_features,
        )[0]

    if args.llm:
        report = generate_llm_report(
            packet,
            model=args.model,
        )
    else:
        report = local_report_from_packet(
            packet
        )

    output_dir = (
        project_root
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transaction_id = packet[
        "transaction_id"
    ]

    packet_path = (
        output_dir
        / f"transaction_{transaction_id}_packet.json"
    )

    report_json_path = (
        output_dir
        / f"transaction_{transaction_id}_report.json"
    )

    report_md_path = (
        output_dir
        / f"transaction_{transaction_id}_report.md"
    )

    with packet_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            packet,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with report_json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    markdown = report_to_markdown(
        report
    )

    with report_md_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            markdown
        )

    print(
        markdown
    )

    print(
        "\nSaved:"
    )
    print(
        packet_path
    )
    print(
        report_json_path
    )
    print(
        report_md_path
    )


if __name__ == "__main__":
    main()
