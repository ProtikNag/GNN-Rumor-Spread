import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy


def count_subnodes(graph, node, depth):
    subnodes = set()

    # Explore neighbors up to the specified depth
    def explore_neighbors(current_node, current_depth):
        if current_depth <= depth:
            for neighbor in graph.neighbors(current_node):
                subnodes.add(neighbor)
                explore_neighbors(neighbor, current_depth + 1)

    # Start exploring from the given node
    explore_neighbors(node, 1)
    return len(subnodes)

def generate_graph(num_nodes):
    # generate weighted and connected graph
    # Graph = nx.gnm_random_graph(num_nodes, (num_nodes*(num_nodes+1)/2)-20, seed=42)
    Graph = nx.full_rary_tree(3, num_nodes)
    source_node = random.randint(0, num_nodes-1)

    # add weights to edges
    # for edge in Graph.edges:
    #     Graph.edges[edge]['weight'] = random.random()

    # add features to nodes
    # node 0 will be the source node
    # each node will have a feature of 3
    # first feature will represent the node's bias (a random value between 0 and 1)
    # second feature will represent if the node is a source node (0 or 1, 1 if the node is the source node)
    # third feature will represent the node's degree
    max_depth = 4
    # Calculating shortest path lengths from the source node to all other nodes
    shortest_paths = nx.shortest_path_length(Graph, source=source_node)

    for node in Graph.nodes:
        Graph.nodes[node]['feature'] = [
            # random.uniform(1, 1),
            1,
            1 if node == source_node else 0,
            Graph.degree[node],
            shortest_paths.get(node, float("inf"))
        ]

    node_features = Graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    edge_index = torch.tensor(list(Graph.edges), dtype=torch.int64).t().contiguous()

    return Graph, node_features, edge_index, source_node


def find_elegible_k_nodes_for_blocking(model_output, source_node, blocked_list, k=1):
    # return the k nodes with maximum probability of being infected
    # if the node with maximum probability is source_node then return the second maximum probability node

    model_output[source_node] = -1  # Exclude the source node
    for node in blocked_list:
        model_output[node] = -1
    model_output = model_output.squeeze()

    return torch.topk(model_output, k).indices.tolist()

def visualize_graph(Graph, source_node, blocked_list, infected_nodes, filename=None):
    # visualize the graph
    # node to remove is black
    # infected nodes are red
    # other nodes are green
    node_colors = []
    for node in Graph.nodes:
        if node in blocked_list:
            node_colors.append('black')
        elif node == source_node:
            node_colors.append('blue')
        elif node in infected_nodes:
            node_colors.append('red')
        else:
            node_colors.append('green')

    nx.draw(Graph, with_labels=True, node_color=node_colors)
    if len(blocked_list) > 0:
        if filename == None:
            plt.savefig("./Figures/graph.pdf")
        else:
            plt.savefig("./Figures/" + filename + ".pdf")
    plt.show()


def visualize_loss(loss_list):
    plt.figure(figsize=(16, 12))
    plt.plot(loss_list)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("./Figures/loss.pdf")
    plt.show()


def simulate_propagation(Graph, source_node):
    # run dfs from source node
    # have a list of infected nodes

    infected_nodes = []
    visited_nodes = []
    propagation_threshhold = 0.5

    def dfs(node, parent):
        if node in visited_nodes or Graph.nodes.data('feature')[node][0] == 0:
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
    # print(num_infected_nodes)
    num_total_nodes = Graph.number_of_nodes()
    proportion_infected_nodes = num_infected_nodes / num_total_nodes

    return proportion_infected_nodes, infected_nodes


def output_with_minimal_infection_rate(Graph, blocked_list, nodes_to_block, source_node=0):
    number_of_nodes = Graph.number_of_nodes()
    minimal_infection_rate = 1
    minimal_infection_rate_output = None
    node_to_block_to_minimize_infection_rate = None
    maximum_viewed_states = 1

    def generate_matrix(graph, number_of_nodes, source, nodes_to_be_blocked, blocked_list=[]):
        matrix = torch.zeros((number_of_nodes, 1), dtype=torch.float)
        shortest_paths = nx.shortest_path_length(graph, source=source)
        sorted_nodes = sorted(shortest_paths, key=lambda x: shortest_paths[x])

        # Assign values in the matrix based on proximity to the source node
        total_blocked = 0
        for idx, node in enumerate(sorted_nodes[1:]):
            if total_blocked == nodes_to_be_blocked:
                break
            if node in blocked_list:
                continue
            matrix[node][0] = 1.0
            total_blocked += 1

        return matrix


    for i in range(maximum_viewed_states):
        output = generate_matrix(
            copy.deepcopy(Graph),
            number_of_nodes,
            source_node,
            nodes_to_block,
            blocked_list
        )

        blocked_nodes = find_elegible_k_nodes_for_blocking(
            copy.deepcopy(output),
            source_node,
            blocked_list,
            k=nodes_to_block
        )
        simulation_graph = copy.deepcopy(Graph)

        for blocked_node in blocked_nodes:
            simulation_graph.nodes[blocked_node]['feature'][0] = 0

        # blocked nodes in the simulation graph should also be 0
        for node in blocked_list:
            simulation_graph.nodes[node]['feature'][0] = 0

        infection_rate, _ = simulate_propagation(simulation_graph, source_node)

        if infection_rate < minimal_infection_rate:
            minimal_infection_rate = infection_rate
            minimal_infection_rate_output = output
            node_to_block_to_minimize_infection_rate = blocked_nodes

    return minimal_infection_rate_output, node_to_block_to_minimize_infection_rate


