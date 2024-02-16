import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize


def visualize_graph(Graph):
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=1)
    node_colors = [cmap(norm(Graph.nodes.data('feature')[node][0])) for node in Graph.nodes]

    for node in Graph.nodes:
        if Graph.nodes.data('feature')[node][0] == -1:
            node_colors[node] = 'black'
    labels = {node: f"{node}\n{Graph.nodes.data('feature')[node][0]:.2f}" for node in Graph.nodes}
    # edge_labels = {(edge[0], edge[1]): Graph[edge[0]][edge[1]]['weight'] for edge in Graph.edges}

    pos = nx.spring_layout(Graph, seed=42)
    nx.draw(
        Graph,
        pos,
        with_labels=True,
        labels=labels,
        nodelist=list(Graph.nodes()),
        node_color=node_colors
    )
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