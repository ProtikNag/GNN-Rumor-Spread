import torch
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_graph(num_nodes):
    # generate weighted and connected graph
    Graph = nx.full_rary_tree(4, num_nodes)

    # add weights to edges
    # for edge in Graph.edges:
    #     Graph.edges[edge]['weight'] = random.random()

    # add features to nodes
    # node 0 will be the source node
    # each node will have a feature of 3
    # first feature will represent the node's bias (a random value between 0 and 1)
    # second feature will represent if the node is a source node (0 or 1, 1 if the node is the source node)
    # third feature will represent the node's degree
    for node in Graph.nodes:
        Graph.nodes[node]['feature'] = [random.random(), 1 if node == 0 else 0, Graph.degree[node]]

    node_features = Graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features])
    edge_index = torch.tensor(list(Graph.edges)).t().contiguous()

    return Graph, node_features, edge_index


def find_elegible_node_for_blocking(model_output, source_node, blocked_list):
    # return the node with maximum probability of being infected
    # if the node with maximum probability is source_node then return the second maximum probability node

    while True:
        probabilities = model_output.squeeze()
        max_prob_node = torch.argmax(probabilities)

        if max_prob_node.item() == source_node or max_prob_node.item() in blocked_list:
            probabilities[max_prob_node] = -1  # Exclude the max prob node
            continue
        else:
            break

    return max_prob_node.item()

def visualize_graph(Graph, source_node, blocked_list, infected_nodes):
    # visualize the graph
    # node to remove is black
    # infected nodes are red
    # other nodes are green
    node_colors = []
    for node in Graph.nodes:
        if node in blocked_list:
            node_colors.append('black')
        elif node == source_node or node in infected_nodes:
            node_colors.append('red')
        else:
            node_colors.append('green')

    nx.draw(Graph, with_labels=True, node_color=node_colors)
    plt.show()


def visualize_loss(loss_list):
    plt.figure(figsize=(16, 12))
    plt.plot(loss_list)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig("./Figures/loss.pdf")
    plt.show()


def simulate_propagation(Graph, source_node):
    # run dfs from source node
    # have a list of infected nodes

    infected_nodes = []
    visited_nodes = []
    propagation_threshhold = 0.5

    def dfs(node, parent):
        if node in visited_nodes:
            return
        visited_nodes.append(node)

        # if node is source then infected
        if parent == -1:
            infected_nodes.append(node)
        elif parent in infected_nodes:
            # if parent is infected then check if node will be infected or not
            probability = Graph.nodes.data('feature')[parent][0]
            infected = 0 if probability <= propagation_threshhold else 1

            if infected:
                infected_nodes.append(node)

        for neighbor in Graph.neighbors(node):
            dfs(neighbor, node)

    # calculate the proportion of infected nodes
    dfs(source_node, -1)
    num_infected_nodes = len(infected_nodes)
    num_total_nodes = Graph.number_of_nodes()
    proportion_infected_nodes = num_infected_nodes / num_total_nodes

    return proportion_infected_nodes, infected_nodes


def output_with_minimal_infection_rate(Graph, blocked_list):
    number_of_nodes = Graph.number_of_nodes()
    minimal_infection_rate = 1
    minimal_infection_rate_output = None
    node_to_block_to_minimize_infection_rate = None

    for i in range(2 * number_of_nodes):
        output = torch.randn(number_of_nodes, 1)
        # make value of source node and blocked nodes 0
        for node in blocked_list:
            output[node] = 0
        output[0] = 0

        blocked_node = find_elegible_node_for_blocking(output, 0, blocked_list)
        simulation_graph = Graph.copy()
        simulation_graph.nodes[blocked_node]['feature'][0] = 0

        infection_rate, _ = simulate_propagation(simulation_graph, 0)

        if infection_rate < minimal_infection_rate:
            minimal_infection_rate = infection_rate
            minimal_infection_rate_output = output
            node_to_block_to_minimize_infection_rate = blocked_node

    return minimal_infection_rate_output, node_to_block_to_minimize_infection_rate


