from graph_utils import generate_graph, update_graph_environment, get_graph_properties

import torch

graph, node_features, edge_index, source_node, edge_weight = generate_graph(10, 'small-world')

for node in graph.nodes:
    for neighbor in graph.neighbors(node):
        print(node, neighbor)
        print(graph[node][neighbor]['weight'])
        print()