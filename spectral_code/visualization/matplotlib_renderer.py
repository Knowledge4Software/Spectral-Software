from __future__ import annotations

import hashlib
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .base import GraphVisualizer


class MatplotlibGraphVisualizer(GraphVisualizer):
    def __init__(
        self,
        layout: str = "spring",
        seed: int = 42,
        dpi: int = 220,
        figure_scale: float = 0.75,
        font_size: int = 8,
        node_size: int = 1800,
        max_label_width: int = 26,
        max_nodes: int = 350,
        export_dot: bool = True,
    ) -> None:
        self.layout = layout
        self.seed = seed
        self.dpi = dpi
        self.figure_scale = figure_scale
        self.font_size = font_size
        self.node_size = node_size
        self.max_label_width = max_label_width
        self.max_nodes = max_nodes
        self.export_dot = export_dot

    def render(
        self,
        graph: nx.Graph,
        *,
        output_dir: str | Path,
        graph_type: str = "graph",
        lang: str = "unknown",
        name: str | None = None,
        title: str | None = None,
        fmt: str = "png",
        **kwargs: Any,
    ) -> dict[str, Path | str | None]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            payload = f"{graph_type}:{lang}:{graph.number_of_nodes()}:{graph.number_of_edges()}"
            name = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

        image_path = output_dir / f"{graph_type}_{lang}_{name}.{fmt.lstrip('.')}"
        dot_path = output_dir / f"{graph_type}_{lang}_{name}.dot" if self.export_dot else None

        if self.export_dot:
            self._try_write_dot(graph, dot_path)

        self._render_matplotlib(graph, image_path, graph_type=graph_type, title=title)
        return {
            "image_path": image_path,
            "dot_path": dot_path,
            "backend": "matplotlib",
            "format": fmt,
        }

    def _try_write_dot(self, graph: nx.Graph, dot_path: Path | None) -> None:
        if dot_path is None:
            return
        try:
            from networkx.drawing.nx_pydot import write_dot
            write_dot(graph, str(dot_path))
        except Exception:
            # If pydot/graphviz is not available, image rendering still proceeds.
            return

    def _layout(self, graph: nx.Graph):
        if graph.number_of_nodes() == 0:
            return {}
        if self.layout == "kamada_kawai":
            return nx.kamada_kawai_layout(graph)
        if self.layout == "circular":
            return nx.circular_layout(graph)
        if self.layout == "shell":
            return nx.shell_layout(graph)
        return nx.spring_layout(graph, seed=self.seed)

    def _render_matplotlib(self, graph: nx.Graph, image_path: Path, *, graph_type: str, title: str | None) -> None:
        G = graph.to_undirected() if graph.is_directed() else graph
        pos = self._layout(G)

        if graph.number_of_nodes() == 0:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=self.dpi)
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
            ax.axis("off")
            fig.savefig(image_path, bbox_inches="tight")
            plt.close(fig)
            return

        if graph.number_of_nodes() > self.max_nodes:
            fig_scale = max(0.6, self.figure_scale * 0.8)
            font_size = max(5, self.font_size - 2)
        else:
            fig_scale = self.figure_scale
            font_size = self.font_size

        color_map = {
            "cfg": "lightcoral",
            "ast": "skyblue",
            "ddg": "lightgreen",
            "pdg": "plum"
        }
        node_color = color_map.get(graph_type, "lightblue")

        figsize = (
            max(8, math.sqrt(graph.number_of_nodes()) * 1.7) * fig_scale,
            max(6, math.sqrt(graph.number_of_nodes()) * 1.1) * fig_scale,
        )
        
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=self.dpi
        )
        
        # Draw Nodes and Edges
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=self.node_size, node_color=node_color, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='dimgray', width=1.0)
        
        # Clean labels and wrap text
        labels = {}
        for node, data in G.nodes(data=True):
            raw_label = str(data.get('label', node)).replace('"', '').split('\\n')[0]
            labels[node] = textwrap.shorten(raw_label, width=self.max_label_width)
            
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)

        ax.set_title(
            title or f"Graph: {graph_type.upper()} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}",
            fontsize=font_size + 4,
            pad=10,
            fontweight="bold"
        )
        ax.axis("off")

        fig.savefig(image_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        node_colors = [self._node_color(data, graph_type) for _, data in G.nodes(data=True)]
        edge_color = "#68717a"

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=graph.is_directed(),
            edge_color=edge_color,
            width=1.2,
            alpha=0.9,
            arrowsize=18,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=int(self.node_size * fig_scale),
            linewidths=1.2,
            edgecolors="#2f3942",
        )

        labels = {node: self._format_label(data, node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size, font_family="DejaVu Sans")

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=max(10, font_size + 4))
        fig.tight_layout()
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

    def _node_color(self, data: dict[str, Any], graph_type: str) -> str:
        if graph_type == "cfg":
            return "#F9D5A7"
        if graph_type == "ast":
            return "#BFD7EA"
        if data.get("block_type"):
            return "#F9D5A7"
        return "#D9EAD3"

    def _format_label(self, data: dict[str, Any], node: Any) -> str:
        label = data.get("label") or data.get("block_type") or str(node)
        label = str(label).replace("\r", " ").replace("\n", " | ")
        label = label.strip()
        if len(label) > self.max_label_width:
            label = textwrap.shorten(label, width=self.max_label_width, placeholder="...")
        return label
