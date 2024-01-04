from GNN import GCN
from utils import (
    simulate_propagation,
    find_elegible_k_nodes_for_blocking,
    generate_graph,
    graph_totally_infected,
    visualize_graph,
    visualize_loss
)
import copy
import torch
import random

input_size = 1           # Number of features per node
hidden_size = 5          # Number of hidden units
output_size = 1          # Size of the output layer

policy = GCN(input_size, hidden_size, output_size)
target = GCN(input_size, hidden_size, output_size)

target.load_state_dict(copy.deepcopy(policy.state_dict()))

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

eps = 1
episode_max = 100

loss_list = []

for episode in range(episode_max):
    print(f"\nStarting episode {episode + 1} of {episode_max}")

    # Generate a random graph with random number of nodes
    num_nodes = random.randint(5, 10)
    Graph1 = generate_graph(num_nodes)
    Graph2 = generate_graph(num_nodes)
    source_node = 0
    infected_nodes = [source_node]
    edge_index1 = torch.tensor(list(Graph1.edges)).t().contiguous()
    node_features1 = torch.randn(num_nodes, input_size)
    edge_index2 = torch.tensor(list(Graph2.edges)).t().contiguous()
    node_features2 = torch.randn(num_nodes, input_size)

    output_target = target(node_features1, edge_index1)
    output_policy = policy(node_features2, edge_index2)

    loss = criterion(output_policy, output_target)

    policy.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item()*3e8)

    if episode % 10 == 0:
        target.load_state_dict(copy.deepcopy(policy.state_dict()))


print("Training complete!")

visualize_loss(loss_list)