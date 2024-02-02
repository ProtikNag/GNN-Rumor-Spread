import random
import torch


def get_greedy_model(graph):
    node_features = graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    node_degrees = node_features[:, 2:3] # [2]: degree
    node_degrees = node_degrees.reshape(-1).tolist()

    for node in node_degrees:
        node = int(node)
        if graph.nodes[node]['feature'][1] == -1:
            graph.nodes[node]['feature'][1] = 0
            break

    return graph


def get_random_model(graph):
    remaining_nodes = []
    for node in graph.nodes:
        if graph.nodes[node]['feature'][1] == -1:
            remaining_nodes.append(node)

    blocked_node = random.choice(remaining_nodes)
    graph.nodes[blocked_node]['feature'][1] = 0

    return graph


def get_trained_model(model, graph):
    node_features = graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.int64).t().contiguous()
    model_output = model(node_features, edge_index)
    blocked_node = torch.argmax(model_output).item()
    graph.nodes[blocked_node]['feature'][1] = 0

    return graph

