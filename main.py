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
        help="Launch the guided terminal workflow.",
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
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Print results without writing checkpoints, metrics, or plots.",
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


def prompt_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    choices_text = "/".join(choices)
    while True:
        answer = input(f"{prompt} [{choices_text}] (default: {default}): ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        print(f"Please choose one of: {', '.join(choices)}")


def prompt_positive_int(prompt: str, default: int) -> int:
    while True:
        answer = input(f"{prompt} (default: {default}): ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            value = 0
        if value > 0:
            return value
        print("Enter a positive integer.")


def prompt_source() -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    print("\nInput options:")
    print("1. Analyze an existing C/C++ source file")
    print("2. Enter C/C++ code directly")
    while True:
        choice = input("Choose input [1/2]: ").strip()
        if choice in {"1", "2"}:
            break
        print("Please enter 1 or 2.")

    if choice == "1":
        while True:
            path = Path(input("Source file path: ").strip().strip('"')).expanduser()
            if path.is_file():
                return path.resolve(), None
            print(f"File not found: {path}")

    print("Enter C/C++ code. Type END on its own line when finished:")
    code = "\n".join(iter(input, "END")) + "\n"
    temporary_directory = tempfile.TemporaryDirectory(prefix="program_analysis_")
    path = Path(temporary_directory.name) / "input.c"
    path.write_text(code, encoding="utf-8")
    return path, temporary_directory


def interactive_arguments() -> argparse.Namespace:
    print("\n=== Program Analysis ===")
    print("Results will be printed in this terminal. No report files will be created.")
    source_path, temporary_directory = prompt_source()
    model = prompt_choice(
        "Model",
        ("multiview", "gcn", "gat", "graphsage"),
        "multiview",
    )
    label = int(prompt_choice("Label (0 = benign, 1 = vulnerable)", ("0", "1"), "0"))
    epochs = prompt_positive_int("Training epochs", 3)
    return argparse.Namespace(
        source=source_path,
        code_file=None,
        code=None,
        interactive=False,
        install_dependencies=False,
        check_dependencies=False,
        compiler_arg=[],
        model=model,
        label=label,
        epochs=epochs,
        hidden_channels=32,
        learning_rate=1e-3,
        batch_size=1,
        device="cpu",
        seed=42,
        skip_training=False,
        output_dir=Path("reports") / "latest_run",
        terminal_only=True,
        temporary_directory=temporary_directory,
    )


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
    temporary_directory = getattr(args, "temporary_directory", temporary_directory)
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
        if not args.terminal_only:
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
        metrics["graph"] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "feature_size": sample.num_node_features,
            "relation_types": len(sample.relation_vocabulary),
        }
        metrics["model"] = args.model
        metrics["source_file"] = str(source_path)
        return metrics
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        args = interactive_arguments()
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
    if not args.terminal_only:
        print(f"Artifacts written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
