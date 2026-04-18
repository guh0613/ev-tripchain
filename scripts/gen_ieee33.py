import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def draw_ieee33():
    G = nx.Graph()

    # 常闭线路（0-based 编号：节点 0—32）
    G.add_edges_from([(i, i + 1) for i in range(0, 17)])       # 主干线 0→17
    G.add_edges_from([(1, 18), (18, 19), (19, 20), (20, 21)])  # 支线 1
    G.add_edges_from([(2, 22), (22, 23), (23, 24)])             # 支线 2
    G.add_edges_from([(5, 25)])
    G.add_edges_from([(i, i + 1) for i in range(25, 32)])       # 支线 3

    # 节点坐标
    pos = {}
    for i in range(0, 18):
        pos[i] = (i, 0)                                         # 主干线 y=0
    for i, node in enumerate(range(18, 22)):
        pos[node] = (1 + i, 1)                                  # 支线 1 y=1
    for i, node in enumerate(range(22, 25)):
        pos[node] = (2 + i, -1)                                 # 支线 2 y=-1
    for i, node in enumerate(range(25, 33)):
        pos[node] = (5 + i, -1)                                 # 支线 3 y=-1

    plt.figure(figsize=(14, 6))

    nx.draw_networkx_nodes(G, pos, node_color='#87CEFA', node_size=350, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color='black', width=2, label="配电线路 (常闭)")

    # 5 条联络线（0-based：20-7, 8-14, 11-21, 17-32, 24-28）
    straight_ties = [(7, 20), (11, 21), (17, 32)]
    curved_ties = [(8, 14), (24, 28)]

    nx.draw_networkx_edges(G, pos, edgelist=straight_ties, edge_color='red',
                           style='dashed', width=1.5, label="联络线 (常开)")
    nx.draw_networkx_edges(G, pos, edgelist=curved_ties, edge_color='red',
                           style='dashed', width=1.5,
                           connectionstyle="arc3,rad=-0.4",
                           arrows=True, arrowstyle='-')

    plt.title("IEEE 33 节点配电网拓扑图", fontsize=18, fontweight='bold')
    plt.axis("off")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig("docs/thesis/pics/fig_3_1.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    draw_ieee33()
