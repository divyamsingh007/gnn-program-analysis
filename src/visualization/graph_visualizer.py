"""Static program-graph visualization helpers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def draw_program_graph(
    graph: nx.Graph,
    output_path: str | Path,
    *,
    title: str = "Program graph",
) -> Path:
    """Render a deterministic spring layout with view-aware node colors."""

    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    positions = nx.spring_layout(graph, seed=42)
    views = [graph.nodes[node].get("view", "unknown") for node in graph.nodes]
    palette = {"ast": "#4c78a8", "cfg": "#f58518", "dataflow": "#54a24b"}
    colors = [palette.get(view, "#bdbdbd") for view in views]
    nx.draw_networkx(graph, positions, ax=axis, node_color=colors, with_labels=False, arrows=True)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
    return target