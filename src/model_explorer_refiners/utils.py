from model_explorer.graph_builder import Graph, GraphNode


def find_node(graph: Graph, node_id: str) -> GraphNode | None:
    """Find a node in a graph.

    Args:
        graph: the graph to search.
        node_id: the id of the node to find.

    Returns:
        The node node which matches the node_id, if found, otherwise None.
    """
    return next(
        (context_node for context_node in graph.nodes if context_node.id == node_id),
        None,
    )
