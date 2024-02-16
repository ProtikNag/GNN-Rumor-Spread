from graph_utils import generate_graph, update_graph_environment, get_graph_properties
from visualization_utils import visualize_graph

graph, node_features, edge_index, source_node, edge_weight = generate_graph(10, 'small-world')

visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)
graph = update_graph_environment(graph)
visualize_graph(graph)