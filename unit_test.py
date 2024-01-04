from utils import generate_graph, simulate_propagation, visualize_graph
import random
import torch
import networkx as nx
import copy

num_nodes = random.randint(10, 10)
Graph, node_features, edge_index, source_node = generate_graph(num_nodes)


def generate_matrix(graph, number_of_nodes, source, nodes_to_be_blocked):
    matrix = torch.zeros((number_of_nodes, 1), dtype=torch.float)
    shortest_paths = nx.shortest_path_length(graph, source=source)
    sorted_nodes = sorted(shortest_paths, key=lambda x: shortest_paths[x])

    print(sorted_nodes)

    # Assign values in the matrix based on proximity to the source node
    for idx, node in enumerate(sorted_nodes[1:nodes_to_be_blocked+1]):
        matrix[node][0] = 1.0

    print(matrix, source, nodes_to_be_blocked)

generate_matrix(copy.deepcopy(Graph), num_nodes, source_node, 4)

print(node_features)

visualize_graph(Graph, source_node, [], [])
ir, infected_nodes = simulate_propagation(Graph, source_node)

for node in infected_nodes:
    Graph.nodes[node]['feature'][0] = 0

visualize_graph(Graph, source_node, [], infected_nodes)