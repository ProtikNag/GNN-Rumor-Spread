import torch
from utils import (
    generate_graph,
    visualize_graph,
    find_elegible_node_for_blocking,
    simulate_propagation
)
from GNN import GCN

torch.manual_seed(42)

input_size = 3           # Number of features per node
hidden_size = 10
output_size = 1
number_of_nodes = 30

model = GCN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("./Models/current_model.pt"))
Graph, node_features, edge_index = generate_graph(number_of_nodes)

output = model(node_features, edge_index)
blocked_node = find_elegible_node_for_blocking(output, 0, [])
Graph.nodes[blocked_node]['feature'][0] = 0
infection_rate, infected_nodes = simulate_propagation(Graph, 0)

visualize_graph(Graph, 0, [blocked_node], infected_nodes)
visualize_graph(Graph, 0, [], infected_nodes)

print(infection_rate)