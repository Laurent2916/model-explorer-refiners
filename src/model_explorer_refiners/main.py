import refiners.fluxion.layers as fl
from model_explorer import Adapter, AdapterMetadata, server
from model_explorer.config import ModelExplorerConfig

from model_explorer_refiners.convert import convert_model


class RefinersAdapter(Adapter):
    metadata = AdapterMetadata(
        id="refiners",
        name="Refiners adapter",
        description="Refiners adapter for Model Explorer",
        source_repo="https://github.com/finegrain/model-explorer-refiners",
    )

    def __init__(self):
        super().__init__()

    @staticmethod
    def visualize(
        model: fl.Chain,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Visualize a refiners model in Model Explorer.

        Args:
            model: the refiners model to visualize.
            host: the host of the server.
            port: the port of the server.
        """
        # construct config
        config = ModelExplorerConfig()
        graph = convert_model(model)
        config.graphs_list.append(graph)
        index = len(config.graphs_list) - 1
        config.model_sources.append({"url": f"graphs://refiners/{index}"})

        # start server
        server.start(
            config=config,
            host=host,
            port=port,
        )
