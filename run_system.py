from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> None:
    print("\nrunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ewaste research-grade pipeline runner")

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="run adaptive deep-learning benchmark")
    p_train.add_argument("--arches", type=str, default="resnet50,vit_b16,convnext_tiny,swin_tiny,efficientnet_b3")
    p_train.add_argument("--epochs", type=int, default=30)

    p_compete = sub.add_parser("compete", help="run deep + traditional model competition")
    p_compete.add_argument("--embedding-arch", type=str, default=None)

    sub.add_parser("cluster", help="run clustering pipeline")
    sub.add_parser("ann", help="run ann hazard pipeline")

    p_all = sub.add_parser("all", help="run full pipeline in sequence")
    p_all.add_argument("--arches", type=str, default="resnet50,vit_b16,convnext_tiny,swin_tiny,efficientnet_b3")
    p_all.add_argument("--epochs", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if args.command == "train":
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "training" / "research_benchmark.py"),
                "--data-dir",
                str(PROJECT_ROOT / "data"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "classification"),
                "--arches",
                args.arches,
                "--epochs",
                str(args.epochs),
            ]
        )
        return

    if args.command == "compete":
        cmd = [
            py,
            str(PROJECT_ROOT / "training" / "model_competition.py"),
            "--data-dir",
            str(PROJECT_ROOT / "data"),
            "--classification-dir",
            str(PROJECT_ROOT / "models" / "classification"),
            "--output-dir",
            str(PROJECT_ROOT / "models" / "competition"),
        ]
        if args.embedding_arch:
            cmd.extend(["--embedding-arch", args.embedding_arch])
        run_cmd(cmd)
        return

    if args.command == "cluster":
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "pipelines" / "clustering_pipeline.py"),
                "--data-dir",
                str(PROJECT_ROOT / "data"),
                "--classification-dir",
                str(PROJECT_ROOT / "models" / "classification"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "clustering"),
            ]
        )
        return

    if args.command == "ann":
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "pipelines" / "ann_hazard_pipeline.py"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "ann"),
            ]
        )
        return

    if args.command == "all":
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "training" / "research_benchmark.py"),
                "--data-dir",
                str(PROJECT_ROOT / "data"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "classification"),
                "--arches",
                args.arches,
                "--epochs",
                str(args.epochs),
            ]
        )

        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "training" / "model_competition.py"),
                "--data-dir",
                str(PROJECT_ROOT / "data"),
                "--classification-dir",
                str(PROJECT_ROOT / "models" / "classification"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "competition"),
            ]
        )

        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "pipelines" / "clustering_pipeline.py"),
                "--data-dir",
                str(PROJECT_ROOT / "data"),
                "--classification-dir",
                str(PROJECT_ROOT / "models" / "classification"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "clustering"),
            ]
        )

        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "pipelines" / "ann_hazard_pipeline.py"),
                "--output-dir",
                str(PROJECT_ROOT / "models" / "ann"),
            ]
        )
        return


if __name__ == "__main__":
    main()
