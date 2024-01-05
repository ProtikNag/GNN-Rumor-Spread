import torch
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from GNN import GCN
from utils import (
    generate_graph,
    simulate_propagation,
    find_elegible_k_nodes_for_blocking
)


def get_trained_model(model, graph, node_features, edge_index, source_node, k):
    model_output = model(node_features, edge_index)
    blocked_nodes = find_elegible_k_nodes_for_blocking(model_output, source_node, [], k)

    for blocked_node in blocked_nodes:
        graph.nodes[blocked_node]['feature'][0] = 0

    infection_rate, infected_nodes = simulate_propagation(copy.deepcopy(graph), source_node)
    return infection_rate, blocked_nodes, infected_nodes


def get_greedy_model(graph, source, k):
    # return output torch with the nodes with the highest degree being 1 and others being 0
    # exclude the source node as 0
    nodes_to_be_blocked = k
    output = torch.zeros((graph.number_of_nodes(), 1))
    degrees = nx.degree(graph)
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    sorted_nodes = [node[0] for node in sorted_nodes]

    for idx, node in enumerate(sorted_nodes):
        if nodes_to_be_blocked == 0:
            break
        if node == source:
            continue
        else:
            output[node] = 1.0
            nodes_to_be_blocked -= 1

    blocked_nodes = find_elegible_k_nodes_for_blocking(output, source, [], k)

    for blocked_node in blocked_nodes:
        graph.nodes[blocked_node]['feature'][0] = 0

    infection_rate, infected_nodes = simulate_propagation(copy.deepcopy(graph), source)
    return infection_rate, blocked_nodes, infected_nodes


def get_random_model(graph, source_node, k):
    # return k random nodes
    # exclude the source node
    # declare output torch with shape number_of_nodes x 1
    output = torch.rand((graph.number_of_nodes(), 1))
    blocked_nodes = find_elegible_k_nodes_for_blocking(output, source_node, [], k)

    for blocked_node in blocked_nodes:
        graph.nodes[blocked_node]['feature'][0] = 0

    infection_rate, infected_nodes = simulate_propagation(copy.deepcopy(graph), source_node)
    return infection_rate, blocked_nodes, infected_nodes


def comparison():
    # three models will be compared:
    # 1. trained model
    # 2. random model
    # 3. greedy model (model that blocks the node with the highest degree)

    input_size = 4  # Number of features per node
    hidden_size = 128
    output_size = 1
    experiment_max = 10
    number_of_nodes = [nodes for nodes in range(10, 51, 5)]
    nodes_to_block = [k for k in range(1, 6, 2)]

    df = pd.DataFrame(columns=[
        'Number of Nodes',
        'Nodes to Block',
        'Trained IR',
        'Max Trained IR',
        'Min Trained IR',
        'Random IR',
        'Max Random IR',
        'Min Random IR',
        'Greedy IR',
        'Max Greedy IR',
        'Min Greedy IR'
    ])

    for num_nodes in number_of_nodes:

        max_trained_infection_rate = 0
        max_random_infection_rate = 0
        max_greedy_infection_rate = 0

        min_trained_infection_rate = 1
        min_random_infection_rate = 1
        min_greedy_infection_rate = 1

        for k in nodes_to_block:

            total_trained_infection_rate = 0
            total_random_infection_rate = 0
            total_greedy_infection_rate = 0

            for i in range(experiment_max):
                Graph, node_features, edge_index, source_node = generate_graph(num_nodes)
                trained_model = GCN(input_size, hidden_size, output_size)
                trained_model.load_state_dict(torch.load("Models/model_v6.pt"))

                trained_infection_rate, trained_blocked_nodes, trained_infected_nodes = get_trained_model(
                    trained_model,
                    copy.deepcopy(Graph),
                    node_features,
                    edge_index,
                    source_node,
                    k
                )

                random_infection_rate, random_blocked_nodes, random_infected_nodes = get_random_model(
                    copy.deepcopy(Graph),
                    source_node,
                    k
                )

                greedy_infection_rate, greedy_blocked_nodes, greedy_infected_nodes = get_greedy_model(
                    copy.deepcopy(Graph),
                    source_node,
                    k
                )

                total_trained_infection_rate += trained_infection_rate
                total_random_infection_rate += random_infection_rate
                total_greedy_infection_rate += greedy_infection_rate

                if trained_infection_rate > max_trained_infection_rate:
                    max_trained_infection_rate = trained_infection_rate
                if random_infection_rate > max_random_infection_rate:
                    max_random_infection_rate = random_infection_rate
                if greedy_infection_rate > max_greedy_infection_rate:
                    max_greedy_infection_rate = greedy_infection_rate

                if trained_infection_rate < min_trained_infection_rate:
                    min_trained_infection_rate = trained_infection_rate
                if random_infection_rate < min_random_infection_rate:
                    min_random_infection_rate = random_infection_rate
                if greedy_infection_rate < min_greedy_infection_rate:
                    min_greedy_infection_rate = greedy_infection_rate


            total_trained_infection_rate /= experiment_max
            total_random_infection_rate /= experiment_max
            total_greedy_infection_rate /= experiment_max

            df.loc[len(df)] = {
                'Number of Nodes': num_nodes,
                'Nodes to Block': k,
                'Trained IR': total_trained_infection_rate,
                'Max Trained IR': max_trained_infection_rate,
                'Min Trained IR': min_trained_infection_rate,
                'Random IR': total_random_infection_rate,
                'Max Random IR': max_random_infection_rate,
                'Min Random IR': min_random_infection_rate,
                'Greedy IR': total_greedy_infection_rate,
                'Max Greedy IR': max_greedy_infection_rate,
                'Min Greedy IR': min_greedy_infection_rate
            }

    df.to_csv('./Comparison/comparison.csv', index=False)

def visualization():
    # from the csv file read the data
    # plot 9 different graphs for each number of nodes
    # each graph will have 3 groups of bars representing the 3 different nodes to block
    # each group will have 3 bars representing the 3 different models

    df = pd.read_csv('./Comparison/comparison.csv')
    number_of_row_column = 3
    # Get unique values for Number of Nodes
    node_values = df['Number of Nodes'].unique()

    # Create subplots for each unique value of 'Number of Nodes'
    fig, axs = plt.subplots(nrows=number_of_row_column, ncols=number_of_row_column, figsize=(16, 12))

    for i, node in enumerate(node_values):
        # Filter DataFrame for each 'Number of Nodes'
        node_df = df[df['Number of Nodes'] == node]

        # Get Nodes to Block and IR values for the current 'Number of Nodes'
        nodes_to_block = node_df['Nodes to Block'].unique()
        trained_ir = list(node_df['Trained IR'])
        random_ir = list(node_df['Random IR'])
        greedy_ir = list(node_df['Greedy IR'])

        # print(trained_ir, random_ir, greedy_ir)

        bar_width = 0.2
        index = np.arange(len(nodes_to_block))

        # Plotting the bar chart for each subplot
        index_i = int(i/number_of_row_column)
        index_j = i%number_of_row_column

        # print(index_i, index_j)

        axs[index_i, index_j].bar(index, trained_ir, bar_width, label='GNN Model')
        axs[index_i, index_j].bar(index + bar_width, random_ir, bar_width, label='Random Model')
        axs[index_i, index_j].bar(index + 2 * bar_width, greedy_ir, bar_width, label='Greedy Model')

        axs[index_i, index_j].set_xlabel('Number of Blocked Nodes')
        axs[index_i, index_j].set_ylabel('Infection Rate')
        axs[index_i, index_j].set_title(f'Number of Nodes: {node}')
        axs[index_i, index_j].set_xticks(index + bar_width)
        axs[index_i, index_j].set_xticklabels(nodes_to_block)
        axs[index_i, index_j].legend()

    plt.tight_layout()
    plt.savefig("./Figures/comparison.pdf")
    plt.show()