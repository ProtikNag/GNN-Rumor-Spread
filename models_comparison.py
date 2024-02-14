import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from tqdm import tqdm

from GNN import GCN
from graph_utils import generate_graph, get_graph_properties, update_graph_environment
from models_utils import get_greedy_model, get_random_model, get_trained_model
from params import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE
)
from params import EXPERIMENT_MAX



def comparison():
    # three models will be compared:
    # 1. trained model
    # 2. random model
    # 3. greedy model (model that blocks the node with the highest degree)

    number_of_nodes = [nodes for nodes in range(10, 151, 20)]

    df = pd.DataFrame(columns=[
        'Number of Nodes',
        'GNN Infection Rate',
        'Random Infection Rate',
        'Max Degree Infection Rate'
    ])

    for num_nodes in tqdm(number_of_nodes):

        total_trained_infection_rate = 0
        total_random_infection_rate = 0
        total_greedy_infection_rate = 0

        for exp in range(EXPERIMENT_MAX):
            Graph, _, _, _, _ = generate_graph(
                num_nodes,
                'small-world'
            )

            trained_model = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            trained_model.load_state_dict(torch.load("Models/model_v10.pt"))

            graph = copy.deepcopy(Graph)

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0:
                    break

                graph = get_trained_model(trained_model, copy.deepcopy(graph))
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_trained = len(infected_nodes) / num_nodes
            total_trained_infection_rate += infection_rate_trained

            # print("Number of Nodes:", num_nodes, "GNN Done")

            graph = copy.deepcopy(Graph)

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0:
                    break

                # print("uninfected: ", uninfected_nodes)
                graph = get_greedy_model(copy.deepcopy(graph))
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_greedy = len(infected_nodes) / num_nodes
            total_greedy_infection_rate += infection_rate_greedy

            graph = copy.deepcopy(Graph)

            # print("Number of Nodes:", num_nodes, "Greedy Done")

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0:
                    break

                graph = get_random_model(copy.deepcopy(graph))
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_random = len(infected_nodes) / num_nodes
            total_random_infection_rate += infection_rate_random

            # print("Number of Nodes:", num_nodes, "Random Done")

        total_trained_infection_rate /= EXPERIMENT_MAX
        total_greedy_infection_rate /= EXPERIMENT_MAX
        total_random_infection_rate /= EXPERIMENT_MAX

        df.loc[len(df)] = {
            'Number of Nodes': num_nodes,
            'GNN Infection Rate': total_trained_infection_rate,
            'Random Infection Rate': total_random_infection_rate,
            'Max Degree Infection Rate': total_greedy_infection_rate,
        }

    df.to_csv('./Comparison/comparison.csv', index=False)

# comparison()

def visualization():
    # from the csv file read the data
    # plot 9 different graphs for each number of nodes
    # each graph will have 3 groups of bars representing the 3 different nodes to block
    # each group will have 3 bars representing the 3 different models

    df = pd.read_csv('./Comparison/comparison.csv')
    df.plot(x="Number of Nodes", y=["GNN Infection Rate", "Random Infection Rate", "Max Degree Infection Rate"], kind="bar")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Infection Rate")
    plt.tight_layout()
    plt.savefig("./Figures/comparison.pdf")
    plt.show()

# comparison()
# visualization()