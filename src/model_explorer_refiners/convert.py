import refiners.fluxion.layers as fl
from model_explorer import ModelExplorerGraphs
from model_explorer.graph_builder import Graph, GraphNode, IncomingEdge, KeyValue

from model_explorer_refiners.utils import find_node


def convert_chain(
    chain: fl.Chain,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners chain layer to nodes.

    Args:
        chain: the chain layer to convert.
        input_nodes: the input nodes of the chain.
    """
    nodes: list[GraphNode] = []
    previous_nodes = input_nodes
    for layer, parent_layer in chain.walk(recurse=False):
        module_nodes, previous_nodes = convert_module(layer, parent_layer, previous_nodes)
        nodes.extend(module_nodes)

    return nodes, previous_nodes


def convert_passthrough(
    passthrough: fl.Passthrough,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners passthrough layer to nodes.

    Args:
        passthrough: the passthrough layer to convert.
        input_nodes: the input nodes of the passthrough.
    """
    nodes, _ = convert_chain(passthrough, input_nodes)
    return nodes, input_nodes


def convert_residual(
    residual: fl.Residual,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners residual layer to nodes.

    Args:
        residual: the residual layer to convert.
        input_nodes: the input nodes of the residual.
    """
    nodes: list[GraphNode] = []
    previous_nodes = input_nodes
    for layer, parent_layer in residual.walk(recurse=False):
        module_nodes, previous_nodes = convert_module(layer, parent_layer, previous_nodes)
        nodes.extend(module_nodes)

    # add a summation node
    summation_node = GraphNode(
        namespace=residual.get_path().replace(".", "/"),
        id=residual.get_path(),
        label="+",
    )
    for node in previous_nodes:
        summation_node.incomingEdges.append(
            IncomingEdge(
                sourceNodeId=node.id,
            ),
        )
    for input_node in input_nodes:
        summation_node.incomingEdges.append(
            IncomingEdge(
                sourceNodeId=input_node.id,
            ),
        )
    nodes.append(summation_node)

    return nodes, [summation_node]


def convert_parallel(
    parallel: fl.Parallel,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners parallel layer to nodes.

    Args:
        parallel: the parallel layer to convert.
        input_nodes: the input nodes of the parallel layer.
    """
    nodes: list[GraphNode] = []
    output_nodes: list[GraphNode] = []
    for layer, parent_layer in parallel.walk(recurse=False):
        module_nodes, module_output_nodes = convert_module(layer, parent_layer, input_nodes)
        nodes.extend(module_nodes)
        output_nodes.extend(module_output_nodes)

    return nodes, output_nodes


def convert_concatenate(
    concatenate: fl.Concatenate,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners concatenate layer to nodes.

    Args:
        concatenate: the concatenate layer to convert.
        input_nodes: the input nodes of the concatenate layer.
    """
    nodes: list[GraphNode] = []
    output_nodes: list[GraphNode] = []
    for layer, parent_layer in concatenate.walk(recurse=False):
        module_nodes, module_output_nodes = convert_module(layer, parent_layer, input_nodes)
        nodes.extend(module_nodes)
        output_nodes.extend(module_output_nodes)

    # add a concatenation node
    concat_node = GraphNode(
        namespace=concatenate.get_path().replace(".", "/"),
        id=concatenate.get_path(),
        label="+",
    )
    for node in output_nodes:
        concat_node.incomingEdges.append(
            IncomingEdge(
                sourceNodeId=node.id,
            ),
        )
    nodes.append(concat_node)

    return nodes, [concat_node]


def convert_distribute(
    distribute: fl.Distribute,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners distribute layer to nodes.

    Args:
        distribute: the distribute layer to convert.
        input_nodes: the input nodes of the distribute layer.
    """
    nodes: list[GraphNode] = []
    output_nodes: list[GraphNode] = []
    for (layer, parent_layer), input_node in zip(distribute.walk(recurse=False), input_nodes, strict=True):
        module_nodes, previous_nodes = convert_module(layer, parent_layer, [input_node])
        nodes.extend(module_nodes)
        output_nodes.extend(previous_nodes)

    return nodes, output_nodes


def convert_other(
    module: fl.Module,
    parent_module: fl.Chain | None,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners module to nodes.

    Args:
        module: the module to convert.
        parent_module: the parent module of the module.
        input_nodes: the input nodes of the module.
    """
    node = GraphNode(
        namespace=parent_module.get_path().replace(".", "/") if parent_module else "",
        id=module.get_path(parent_module),
        label=module._get_name(),  # type: ignore
        attrs=[
            KeyValue(
                key=key,
                value=str(value),
            )
            for key, value in module.basic_attributes().items()
        ],
    )
    setattr(node, "refiners_module", module)  # noqa: B010
    for input_node in input_nodes:
        node.incomingEdges.append(
            IncomingEdge(
                sourceNodeId=input_node.id,
            ),
        )
    return [node], [node]


def convert_module(  # type: ignore
    module: fl.Module,
    parent_module: fl.Chain | None,
    input_nodes: list[GraphNode],
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Convert a refiners module to nodes.

    Args:
        module: the module to convert.
        parent_module: the parent module of the module.
        input_nodes: the input nodes of the module.
    """
    match module:
        case fl.Parallel():
            return convert_parallel(module, input_nodes)
        case fl.Distribute():
            return convert_distribute(module, input_nodes)
        case fl.Concatenate():
            return convert_concatenate(module, input_nodes)
        case fl.Residual():
            return convert_residual(module, input_nodes)
        case fl.Passthrough():
            return convert_passthrough(module, input_nodes)
        case fl.Chain():
            return convert_chain(module, input_nodes)
        case fl.UseContext():
            return convert_other(module, parent_module=parent_module, input_nodes=[])
        case fl.Parameter():
            return convert_other(module, parent_module=parent_module, input_nodes=[])
        case _:
            return convert_other(module, parent_module, input_nodes)


def convert_model(model: fl.Chain) -> ModelExplorerGraphs:
    """Convert a refiners model to a Graph.

    Args:
        model: the model to convert.
    """
    # initialize the graph, with an input and output node
    graph = Graph(id=model._get_name())  # type: ignore
    input_node = GraphNode(id="input", label="input")
    graph.nodes.append(input_node)
    output_node = GraphNode(id="output", label="output")
    graph.nodes.append(output_node)

    # convert the model
    model_nodes, model_output_nodes = convert_module(
        module=model,
        parent_module=None,
        input_nodes=[input_node],
    )
    graph.nodes.extend(model_nodes)

    # connect the model's output to the output node
    for node in model_output_nodes:
        output_node.incomingEdges.append(
            IncomingEdge(
                sourceNodeId=node.id,
            ),
        )

    # connect the model's context nodes
    for node in graph.nodes:
        refiners_module = getattr(node, "refiners_module", None)
        if isinstance(refiners_module, fl.SetContext):
            # find the context node
            context_node = find_node(
                graph=graph,
                node_id=f"context.{refiners_module.context}.{refiners_module.key}",
            )

            # if it doesn't exist, create it
            if context_node is None:
                context_node = GraphNode(
                    namespace=f"context/{refiners_module.context}/{refiners_module.key}",
                    id=f"context.{refiners_module.context}.{refiners_module.key}",
                    label=refiners_module.key,
                )

            # connect the nodes: node -> context_node
            context_node.incomingEdges.append(
                IncomingEdge(
                    sourceNodeId=node.id,
                ),
            )
            graph.nodes.append(context_node)
        if isinstance(refiners_module, fl.UseContext):
            # find the context node
            context_node = find_node(
                graph=graph,
                node_id=f"context.{refiners_module.context}.{refiners_module.key}",
            )

            # if it doesn't exist, create it
            if context_node is None:
                context_node = GraphNode(
                    namespace=f"context/{refiners_module.context}/{refiners_module.key}",
                    id=f"context.{refiners_module.context}.{refiners_module.key}",
                    label=refiners_module.key,
                )
                graph.nodes.append(context_node)

            # connect the nodes: context_node -> node
            node.incomingEdges.append(
                IncomingEdge(
                    sourceNodeId=f"context.{refiners_module.context}.{refiners_module.key}",
                ),
            )

    return ModelExplorerGraphs(graphs=[graph])
