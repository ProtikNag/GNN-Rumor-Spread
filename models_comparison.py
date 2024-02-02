import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
import random as random

from GNN import GCN
from graph_utils import generate_graph, get_graph_properties, update_graph_environment
from models_utils import get_greedy_model, get_random_model, get_trained_model
from params import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE
)



def comparison():
    # three models will be compared:
    # 1. trained model
    # 2. random model
    # 3. greedy model (model that blocks the node with the highest degree)

    experiment_max = 20
    number_of_nodes = [nodes for nodes in range(10, 51, 5)]

    df = pd.DataFrame(columns=[
        'Number of Nodes',
        'Trained IR',
        'Random IR',
        'Max Degree IR'
    ])

    for num_nodes in number_of_nodes:

        total_trained_infection_rate = 0
        total_random_infection_rate = 0
        total_greedy_infection_rate = 0

        for exp in range(experiment_max):

            edge_creation_probability = random.uniform(0.25, 0.3)
            Graph, _, _, _ = generate_graph(
                num_nodes,
                edge_creation_probability,
                'tree'
            )

            trained_model = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            trained_model.load_state_dict(torch.load("Models/model_v7.pt"))

            previous_infected_nodes = 0
            graph = copy.deepcopy(Graph)

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0 or len(infected_nodes) == previous_infected_nodes:
                    break
                previous_infected_nodes = len(infected_nodes)

                graph = get_trained_model(
                    trained_model,
                    copy.deepcopy(graph)
                )
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_trained = len(infected_nodes) / num_nodes
            total_trained_infection_rate += infection_rate_trained

            graph = copy.deepcopy(Graph)
            previous_infected_nodes = 0

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0 or len(infected_nodes) == previous_infected_nodes:
                    break
                previous_infected_nodes = len(infected_nodes)

                graph = get_greedy_model(copy.deepcopy(graph))
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_greedy = len(infected_nodes) / num_nodes
            total_greedy_infection_rate += infection_rate_greedy

            graph = copy.deepcopy(Graph)
            previous_infected_nodes = 0

            while True:
                _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
                if len(uninfected_nodes) == 0 or len(infected_nodes) == previous_infected_nodes:
                    break
                previous_infected_nodes = len(infected_nodes)

                graph = get_random_model(copy.deepcopy(graph))
                graph = update_graph_environment(copy.deepcopy(graph))

            _, infected_nodes, _, _ = get_graph_properties(graph)
            infection_rate_random = len(infected_nodes) / num_nodes
            total_random_infection_rate += infection_rate_random

        total_trained_infection_rate /= experiment_max
        total_greedy_infection_rate /= experiment_max
        total_random_infection_rate /= experiment_max

        df.loc[len(df)] = {
            'Number of Nodes': num_nodes,
            'Trained IR': total_trained_infection_rate,
            'Random IR': total_random_infection_rate,
            'Max Degree IR': total_greedy_infection_rate,
        }

    df.to_csv('./Comparison/comparison.csv', index=False)

# comparison()

def visualization():
    # from the csv file read the data
    # plot 9 different graphs for each number of nodes
    # each graph will have 3 groups of bars representing the 3 different nodes to block
    # each group will have 3 bars representing the 3 different models

    df = pd.read_csv('./Comparison/comparison.csv')
    df.plot(x="Number of Nodes", y=["Trained IR", "Random IR", "Max Degree IR"], kind="bar")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Infection Rate")
    plt.tight_layout()
    plt.savefig("./Figures/comparison.pdf")
    plt.show()

# visualization()