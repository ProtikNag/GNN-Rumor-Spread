import torch

from graph_utils import (
    generate_graph,
    update_graph_environment,
    get_graph_properties
)
from visualization_utils import visualize_graph, visualize_loss
from GNN import GCN
from params import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    LEARNING_RATE
)
from models_utils import find_node_to_block


def find_new_source_nodes(Graph):
    # if a node is infected, its immediate neighbors will be also infected

    infected_nodes = []
    for node in Graph.nodes:
        if Graph.nodes.data('feature')[node][0] == 1:
            infected_nodes.append(node)

    new_source_nodes = []
    for node in infected_nodes:
        for neighbor in Graph.neighbors(node):
            if Graph.nodes.data('feature')[neighbor][0] == -1:
                new_source_nodes.append(neighbor)
                Graph.nodes[neighbor]['feature'][0] = 1

    return Graph, new_source_nodes



Graph, node_features, edge_index, source_node, edge_weight = generate_graph(10, 'small-world')

# source_node = 4
trained_model = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
trained_model.load_state_dict(torch.load("Models/model_v10.pt"))

total_nodes_to_be_blocked = 10

blocked_nodes = []
infected_nodes = [source_node]

for i in range(total_nodes_to_be_blocked):
    _, _, _, uninfected_nodes = get_graph_properties(Graph)

    if len(uninfected_nodes) == 0:
        break

    visualize_graph(Graph)

    model_output = trained_model(node_features, edge_index, edge_weight)
    blocked_node = find_node_to_block(model_output, infected_nodes, blocked_nodes)
    blocked_nodes.append(blocked_node)

    Graph.nodes[blocked_node]['feature'][0] = -1
    Graph = update_graph_environment(Graph)
    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    node_features = Graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)

print("Blocked nodes: ", blocked_nodes)