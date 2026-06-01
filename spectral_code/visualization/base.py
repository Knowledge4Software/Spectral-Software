from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import networkx as nx


class GraphVisualizer(ABC):
    @abstractmethod
    def render(self, graph: nx.Graph, **kwargs: Any) -> dict[str, Path | str | None]:
        raise NotImplementedError
