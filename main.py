import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from GNN import GCN
from utils import (
    generate_graph,
    output_with_minimal_infection_rate,
    visualize_loss,
    simulate_propagation
)
from tqdm import tqdm
import copy


if __name__ == '__main__':
    torch.manual_seed(42)

    # Hyperparameters
    episode_max = 100
    input_size = 4           # Number of features per node
    hidden_size = 128
    output_size = 1
    learning_rate = 0.001

    # Initialize the policy
    policy = GCN(input_size, hidden_size, output_size)

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_list = []                                              # List to store the loss values
    infection_rate = []                                         # List to store the infection rate values

    for episode in tqdm(range(episode_max)):
        # Generate a random graph
        num_nodes = random.randint(50, 50)
        Graph, node_features, edge_index, source_node = generate_graph(num_nodes)

        already_blocked_list = []
        min_infection_rate = 1
        total_loss = 0

        for i in range(num_nodes-2):
            # remaining_nodes = num_nodes - len(already_blocked_list)
            # nodes_to_block = int(np.sqrt(remaining_nodes))
            nodes_to_block = 8

            target_output, blocked_nodes = output_with_minimal_infection_rate(
                copy.deepcopy(Graph),
                already_blocked_list,
                nodes_to_block,
                source_node
            )

            policy_output = policy(node_features, edge_index)

            policy.train()
            optimizer.zero_grad()
            loss = criterion(policy_output, target_output)
            loss.backward()
            optimizer.step()

            # print(policy_output, target_output, loss.item())

            already_blocked_list.extend(blocked_nodes)
            total_loss += loss.item()

            # Blocked node cannot propagate the infection
            for blocked_node in blocked_nodes:
                Graph.nodes[blocked_node]['feature'][0] = 0

            ir, _ = simulate_propagation(copy.deepcopy(Graph), 0)
            infection_rate.append(ir)

        total_loss /= num_nodes
        loss_list.append(total_loss)

    visualize_loss(loss_list)
    # visualize_loss(infection_rate)

    print("\nTraining completed!")
    print("Saving models...")
    current_model_path = "Models/model_v5.pt"
    torch.save(policy.state_dict(), current_model_path)
    print("Models saved successfully!")

