"""Single-command entry point for the program-analysis pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REQUIRED_MODULES = {
    "clang.cindex": "libclang",
    "networkx": "networkx",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, graph, train, evaluate, and visualize a C/C++ program."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="Path to a C or C++ source file.")
    source.add_argument("--code-file", type=Path, help="Alias for --source.")
    source.add_argument(
        "--code",
        help="Inline C/C++ source code. Use --interactive for multiline input.",
    )
    source.add_argument(
        "--interactive",
        action="store_true",
        help="Read source code until a line containing only END.",
    )
    parser.add_argument(
        "--install-dependencies",
        action="store_true",
        help="Install requirements.txt with the current Python interpreter before running.",
    )
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Check required imports and exit.",
    )
    parser.add_argument(
        "--compiler-arg",
        action="append",
        default=[],
        help="Compiler argument passed to each Clang extractor; repeat as needed.",
    )
    parser.add_argument(
        "--model",
        choices=("multiview", "gcn", "gat", "graphsage"),
        default="multiview",
        help="GNN architecture (default: multiview).",
    )
    parser.add_argument("--label", type=int, choices=(0, 1), default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="Torch device, such as cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Build and evaluate the graph with an untrained model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "latest_run",
        help="Directory for metrics, checkpoint, and plots.",
    )
    return parser


def install_dependencies() -> None:
    requirements = Path(__file__).with_name("requirements.txt")
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    subprocess.run(command, check=True)


def check_dependencies() -> list[str]:
    missing: list[str] = []
    for module, package in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as error:
            missing.append(f"{package}: {error}")
    return missing


def read_source(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.source or args.code_file:
        path = (args.source or args.code_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        return path, None

    if args.code is not None:
        code = args.code
    elif args.interactive:
        print("Enter C/C++ code. Type END on its own line to finish:")
        code = "\n".join(iter(input, "END")) + "\n"
    else:
        raise ValueError("Provide --source, --code-file, --code, or --interactive")

    temporary_directory = tempfile.TemporaryDirectory(prefix="program_analysis_")
    path = Path(temporary_directory.name) / "input.c"
    path.write_text(code, encoding="utf-8")
    return path, temporary_directory


def create_model(name: str, feature_size: int, relation_count: int, hidden_channels: int):
    if name == "gcn":
        from src.models.gcn import GCNClassifier

        return GCNClassifier(feature_size, hidden_channels=hidden_channels)
    if name == "gat":
        from src.models.gat import GATClassifier

        return GATClassifier(feature_size, hidden_channels=hidden_channels)
    if name == "graphsage":
        from src.models.graphsage import GraphSAGEClassifier

        return GraphSAGEClassifier(feature_size, hidden_channels=hidden_channels)

    from src.models.multiview_gnn import MultiViewGNN

    return MultiViewGNN(
        feature_size,
        num_relations=relation_count,
        hidden_channels=hidden_channels,
    )


def run_pipeline(args: argparse.Namespace) -> dict:
    import torch

    from src.datasets.dataset_loader import create_graph_loader
    from src.datasets.preprocessing import batch_graphs_to_data
    from src.evaluation.evaluate import evaluate_model
    from src.extraction.ast_extractor import extract_ast
    from src.extraction.cfg_extractor import extract_cfg
    from src.extraction.dataflow_extractor import extract_dataflow
    from src.features.node_features import encode_graph_nodes, fit_node_feature_encoder
    from src.graphs.graph_builder import build_multiview_graph
    from src.training.checkpoints import save_checkpoint
    from src.training.train import train_model
    from src.visualization.graph_visualizer import draw_program_graph
    from src.visualization.result_plots import plot_confusion_matrix
    from src.visualization.training_plots import plot_training_history

    source_path, temporary_directory = read_source(args)
    try:
        torch.manual_seed(args.seed)
        ast_payload = extract_ast(source_path, args.compiler_arg)
        cfg_payload = extract_cfg(source_path, args.compiler_arg)
        dataflow_payload = extract_dataflow(source_path, args.compiler_arg)
        graph = build_multiview_graph(ast_payload, cfg_payload, dataflow_payload)
        encoder = fit_node_feature_encoder(graph)
        graph = encode_graph_nodes(graph, encoder)
        samples = batch_graphs_to_data([graph], labels=[args.label])
        loader = create_graph_loader(samples, batch_size=args.batch_size)
        sample = samples[0]
        model = create_model(
            args.model,
            sample.num_node_features,
            len(sample.relation_vocabulary),
            args.hidden_channels,
        )
        history = {"train_loss": []}
        if not args.skip_training:
            history = train_model(
                model,
                loader,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                device=args.device,
            )
        metrics = evaluate_model(model, loader, device=args.device)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            args.output_dir / "model.pt",
            model,
            epoch=args.epochs if not args.skip_training else 0,
            history=history,
        )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "source_file": str(source_path),
                    "model": args.model,
                    "label": args.label,
                    "graph": {
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "feature_size": sample.num_node_features,
                        "relation_types": len(sample.relation_vocabulary),
                    },
                    "history": history,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if history["train_loss"]:
            plot_training_history(history, args.output_dir / "training.png")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            args.output_dir / "confusion_matrix.png",
            class_names=["benign", "vulnerable"],
        )
        draw_program_graph(graph, args.output_dir / "program_graph.png")
        return metrics
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.install_dependencies:
        install_dependencies()
    missing = check_dependencies()
    if missing:
        parser.error(
            "Missing required dependencies. Run with --install-dependencies or install them manually:\n"
            + "\n".join(missing)
        )
    if args.check_dependencies:
        print("All required dependencies are available.")
        return 0
    if args.epochs < 1 or args.hidden_channels < 1 or args.learning_rate <= 0:
        parser.error("--epochs, --hidden-channels, and --learning-rate must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    try:
        metrics = run_pipeline(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
