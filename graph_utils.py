import torch
import networkx as nx
import random
import copy


def update_graph_environment(Graph):
    # update the graph environment
    # source nodes will infect all its neighbors
    # blocked nodes cannot be infected
    infected_nodes = []
    for node in Graph.nodes:
        if Graph.nodes.data('feature')[node][0] == 1:
            infected_nodes.append(node)

    updated_this_iteration = []              # a node can be updated through only one source node
    increment_factor = 0.25
    for node in infected_nodes:
        for neighbor in Graph.neighbors(node):
            if neighbor in updated_this_iteration:
                continue
            if Graph.nodes.data('feature')[neighbor][0] != -1:
                Graph.nodes[neighbor]['feature'][0] = min(1, (Graph.nodes[neighbor]['feature'][0] + increment_factor))
                updated_this_iteration.append(neighbor)

    # degree of the nodes will be updated
    # if node is connected to a blocked node or source node, its degree will be lower than actual degree
    for node in Graph.nodes:
        Graph.nodes[node]['feature'][1] = Graph.degree[node]

        for neighbor in Graph.neighbors(node):
            if Graph.nodes.data('feature')[neighbor][0] in [-1, 1]:
                Graph.nodes[node]['feature'][1] -= 1

    def bfs(Graph, current_node):
        # if the path is through a blocked node, we need to check if there is another shortest path
        # if there is no other shortest path, the shortest path length will be updated to 20
        visited_nodes = set()
        visited_nodes.add(current_node)
        queue = [current_node]
        distance = dict()
        distance[current_node] = 0
        shortest_path_length = Graph.number_of_nodes()

        while queue:
            node = queue.pop(0)
            if Graph.nodes.data('feature')[node][0] == 1:
                if distance[node] < shortest_path_length:
                    shortest_path_length = distance[node]
                continue
            for neighbor in Graph.neighbors(node):
                if neighbor not in visited_nodes and Graph.nodes.data('feature')[neighbor][0] != -1:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
        return shortest_path_length

    # calculating the shortest path from each node using a bfs
    for node in Graph.nodes:
        if Graph.nodes[node]['feature'][0] == 1:
            Graph.nodes[node]['feature'][2] = 0
            continue
        elif Graph.nodes[node]['feature'][0] == -1:
            Graph.nodes[node]['feature'][2] = Graph.number_of_nodes()
        Graph.nodes[node]['feature'][2] = bfs(copy.deepcopy(Graph), node)

    return Graph

def get_graph_properties(Graph):
    """
    :param Graph: NetworkX Graph
    :return: number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes
    """

    infected_nodes = []
    blocked_nodes = []
    uninfected_nodes = []
    number_of_nodes = Graph.number_of_nodes()

    # -1: uninfected
    # 0: blocked
    # 1: infected
    for node in Graph.nodes:
        if Graph.nodes.data('feature')[node][0] not in [-1, 1]:
            uninfected_nodes.append(node)
        elif Graph.nodes.data('feature')[node][0] == -1:
            blocked_nodes.append(node)
        elif Graph.nodes.data('feature')[node][0] == 1:
            infected_nodes.append(node)

    return number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes

def generate_graph(num_nodes, graph_type="erdos_renyi"):
    Graph = None

    if graph_type == 'small-world':
        k = 3       # everage degree of the nodes
        p = 0.8     # probability of rewiring
        Graph = nx.watts_strogatz_graph(num_nodes, k, p)
    elif graph_type == 'erdos_renyi':
        edge_creation_probability = random.uniform(0.15, 0.3)
        Graph = nx.fast_gnp_random_graph(num_nodes, edge_creation_probability, seed=42)

    source_node = random.randint(0, num_nodes-1)
    # add weights to edges
    # for edge in Graph.edges:
    #     Graph.edges[edge]['weight'] = random.random()

    # add 4 features to each node
    # [0]: opinion value, [1]: degree, [2]: shortest path length from the source node
    shortest_paths = nx.shortest_path_length(Graph, source=source_node)

    for node in Graph.nodes:
        Graph.nodes[node]['feature'] = [
            1 if node == source_node else 0,
            Graph.degree[node],
            shortest_paths.get(node, Graph.number_of_nodes())
        ]

    node_features = Graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    edge_index = torch.tensor(list(Graph.edges), dtype=torch.int64).t().contiguous()

    return Graph, node_features, edge_index, source_node


def simulate_propagation(Graph):
    # run dfs from every infected nodes
    # have a list of infected nodes

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    visited_nodes = set()

    def dfs(node, parent):
        if (node in visited_nodes or Graph.nodes.data('feature')[node][0] == -1):
            return
        visited_nodes.add(node)

        if parent in infected_nodes:
            if node not in infected_nodes:
                infected_nodes.append(node)

        for neighbor in Graph.neighbors(node):
            dfs(neighbor, node)

    # calculate the proportion of infected nodes
    for source_node in infected_nodes:
        if source_node not in visited_nodes:
            dfs(source_node, -1)
    num_infected_nodes = len(set(infected_nodes))
    proportion_infected_nodes = num_infected_nodes / number_of_nodes

    return proportion_infected_nodes


def output_with_minimal_infection_rate(Graph):
    _, _, _, uninfected_nodes = get_graph_properties(copy.deepcopy(Graph))
    minimal_infection_rate = 1
    minimal_infection_rate_output = None
    node_to_block_to_minimize_infection_rate = None

    for node in uninfected_nodes:
        simulation_graph = copy.deepcopy(Graph)
        simulation_graph.nodes[node]['feature'][0] = -1

        infection_rate = simulate_propagation(copy.deepcopy(simulation_graph))

        if infection_rate < minimal_infection_rate:
            minimal_infection_rate = infection_rate
            minimal_infection_rate_output = torch.zeros((Graph.number_of_nodes(), 1), requires_grad=False)
            minimal_infection_rate_output[node] = 1
            node_to_block_to_minimize_infection_rate = node

    return minimal_infection_rate_output, node_to_block_to_minimize_infection_rate