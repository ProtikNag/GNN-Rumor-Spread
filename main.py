import random

import torch
import torch.nn as nn
import torch.optim as optim
from GNN import GCN
from utils import (
    generate_graph,
    output_with_minimal_infection_rate,
    visualize_loss,
)


if __name__ == '__main__':
    torch.manual_seed(42)

    # Hyperparameters
    episode_max = 100
    input_size = 3           # Number of features per node
    hidden_size = 10
    output_size = 1
    learning_rate = 0.0001

    # Initialize the policy
    policy = GCN(input_size, hidden_size, output_size)

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_list = []                                              # List to store the loss values

    for episode in range(episode_max):
        if episode % 10 == 0:
            print(f"Starting episode {episode} of {episode_max}")

        # Generate a random graph
        num_nodes = random.randint(15, 15)
        Graph, node_features, edge_index = generate_graph(num_nodes)

        already_blocked_list = []
        min_infection_rate = 1
        total_loss = 0


        for i in range(num_nodes-5):
            target_output, blocked_node = output_with_minimal_infection_rate(Graph, already_blocked_list)
            policy_output = policy(node_features, edge_index)

            policy.train()
            optimizer.zero_grad()
            loss = criterion(policy_output, target_output)
            loss.backward()
            optimizer.step()

            already_blocked_list.append(blocked_node)
            total_loss += loss.item()

            # Blocked node cannot propagate the infection
            Graph.nodes[blocked_node]['feature'][0] = 0

        total_loss /= num_nodes
        loss_list.append(total_loss)

    visualize_loss(loss_list)

    print("\nTraining completed!")
    print("Saving models...")
    current_model_path = "./Models/current_model.pt"
    torch.save(policy.state_dict(), current_model_path)
    print("Models saved successfully!")

