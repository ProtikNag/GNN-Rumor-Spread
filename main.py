import random
import torch
import torch.nn as nn
import torch.optim as optim
from GNN import GCN
from tqdm import tqdm
import copy


from graph_utils import (
    generate_graph,
    output_with_minimal_infection_rate,
    get_graph_properties,
    update_graph_environment
)
from visualization_utils import visualize_loss, visualize_graph
from models_comparison import comparison, visualization
from params import (
    EPISODE_MAX,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,   
    LEARNING_RATE
)


if __name__ == '__main__':
    torch.manual_seed(42)

    # Initialize the policy
    policy = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    loss_list = []                                              # List to store the loss values

    for episode in tqdm(range(EPISODE_MAX)):
        # Generate a random graph
        num_nodes = random.randint(100, 100)
        Graph, node_features, edge_index, source_node, edge_weight = generate_graph(
            num_nodes,
            'small-world'
        )

        previous_infected_nodes = 0
        total_loss = 0
        iteration = 0

        while True:
            _, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
            # visualize_graph(Graph, blocked_nodes, infected_nodes)

            if len(uninfected_nodes) == 0:
                break

            target_output, blocked_node = output_with_minimal_infection_rate(copy.deepcopy(Graph))
            policy_output = policy(node_features, edge_index, edge_weight)

            policy.train()
            loss = criterion(policy_output, target_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Blocked node cannot propagate the infection
            # print(Graph.nodes.data('feature'))
            Graph.nodes[blocked_node]['feature'][0] = -1
            Graph = update_graph_environment(copy.deepcopy(Graph))
            # print(Graph.nodes.data('feature'))

            # if previous_infected_nodes == len(infected_nodes):
            #     break
            # previous_infected_nodes = len(infected_nodes)

            iteration += 1

        total_loss /= iteration
        loss_list.append(total_loss)

    visualize_loss(loss_list)

    print("\nTraining completed!")
    print("Saving models...")
    current_model_path = "Models/model_v10.pt"
    torch.save(policy.state_dict(), current_model_path)
    print("Models saved successfully!")

    comparison()
    visualization()



