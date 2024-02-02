from matplotlib import pyplot as plt
import networkx as nx


def visualize_graph(Graph, blocked_list, infected_nodes, filename=None):
    # visualize the graph
    # node to remove is black
    # infected nodes are red
    # other nodes are green
    node_colors = []
    for node in Graph.nodes:
        if node in blocked_list:
            node_colors.append('black')
        elif node in infected_nodes:
            node_colors.append('red')
        else:
            node_colors.append('green')

    pos = nx.spring_layout(Graph, seed=42)
    nx.draw(Graph, pos, with_labels=True, node_color=node_colors)
    if len(blocked_list) > 0:
        if filename == None:
            plt.savefig("./Figures/graph.pdf")
        else:
            plt.savefig("./Figures/" + filename + ".pdf")
    plt.savefig("./Figures/graph.pdf")
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