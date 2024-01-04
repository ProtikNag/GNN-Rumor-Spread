import copy

import torch
from utils import (
    generate_graph,
    visualize_graph,
    find_elegible_k_nodes_for_blocking,
    simulate_propagation
)
from GNN import GCN
import matplotlib.pyplot as plt

torch.manual_seed(42)


def visualize_trained_vs_untrained(trained_infection_rate, untrained_infection_rate):
    # visualize the graph
    index = [i for i in range(10)]
    bar_width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(index, trained_infection_rate, bar_width, label='Trained')
    plt.bar([i + bar_width for i in index], untrained_infection_rate, bar_width, label='Random')

    plt.xlabel('Case')
    plt.ylabel('Infection Rate')
    plt.title('Performance comparison between trained and random models on random graphs with 25 nodes')
    plt.xticks([i + bar_width / 2 for i in index], index)
    plt.legend()

    plt.tight_layout()
    plt.savefig("./Figures/trained_vs_random.pdf")
    plt.show()


def test(model, graph, node_features, edge_index, source_node, k):
    output = model(node_features, edge_index)
    print(output)
    blocked_nodes = find_elegible_k_nodes_for_blocking(output, source_node, [], k)

    for blocked_node in blocked_nodes:
        graph.nodes[blocked_node]['feature'][0] = 0

    infection_rate, infected_nodes = simulate_propagation(copy.deepcopy(graph), source_node)

    return infection_rate, blocked_nodes, infected_nodes


def random_choice(graph, source_node, k):
    output = torch.rand((25, 1))
    blocked_nodes = find_elegible_k_nodes_for_blocking(output, source_node, [], k)

    for blocked_node in blocked_nodes:
        graph.nodes[blocked_node]['feature'][0] = 0

    infection_rate, infected_nodes = simulate_propagation(copy.deepcopy(graph), source_node)

    return infection_rate, blocked_nodes, infected_nodes


input_size = 4           # Number of features per node
hidden_size = 128
output_size = 1
number_of_nodes = 25
nodes_to_block = 5
trained_ir = []
random_ir = []

trained_model = GCN(input_size, hidden_size, output_size)
trained_model.load_state_dict(torch.load("Models/model_v5.pt"))

for i in range(10):
    Graph, node_features, edge_index, source_node = generate_graph(number_of_nodes)
    trained_infection_rate, trained_blocked_nodes, trained_infected_nodes = test(
        trained_model,
        copy.deepcopy(Graph),
        node_features,
        edge_index,
        source_node,
        nodes_to_block
    )
    random_infection_rate, random_blocked_nodes, random_infected_nodes = random_choice(
        copy.deepcopy(Graph),
        source_node,
        nodes_to_block
    )
    trained_ir.append(trained_infection_rate)
    random_ir.append(random_infection_rate)

visualize_trained_vs_untrained(trained_ir, random_ir)


Graph, node_features, edge_index, source_node = generate_graph(number_of_nodes)
trained_infection_rate, trained_blocked_nodes, trained_infected_nodes = test(
    trained_model,
    copy.deepcopy(Graph),
    node_features,
    edge_index,
    source_node,
    nodes_to_block
)
random_infection_rate, random_blocked_nodes, random_infected_nodes = random_choice(
    copy.deepcopy(Graph),
    source_node,
    nodes_to_block
)

visualize_graph(copy.deepcopy(Graph), source_node, random_blocked_nodes, random_infected_nodes, filename='random_graph')
visualize_graph(copy.deepcopy(Graph), source_node, trained_blocked_nodes, trained_infected_nodes, filename='trained_graph')



