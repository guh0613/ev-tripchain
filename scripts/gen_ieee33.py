import networkx as nx
import matplotlib.pyplot as plt

# 关键：设置中文字体，防止中文显示为方块
# 这里同时写入了 Windows (SimHei/Microsoft YaHei) 和 Mac (Songti SC/Heiti TC) 的常见字体，系统会自动匹配存在的那个
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False  # 保证坐标轴负号正常显示

def draw_chinese_ieee33():
    G = nx.Graph()
    
    # 1. 添加常闭线路 (实线)
    G.add_edges_from([(i, i+1) for i in range(1, 18)]) # 主干线
    G.add_edges_from([(2, 19), (19, 20), (20, 21), (21, 22)]) # 支线 1
    G.add_edges_from([(3, 23), (23, 24), (24, 25)]) # 支线 2
    G.add_edges_from([(6, 26)])
    G.add_edges_from([(i, i+1) for i in range(26, 33)]) # 支线 3
    
    # 2. 重新设计绝对不会交叉和穿模的物理坐标
    pos = {}
    for i in range(1, 19): pos[i] = (i, 0)                  # 主干线 (y=0)
    for i, node in enumerate(range(19, 23)): pos[node] = (2 + i, 1)   # 支线 1：向上翻折 (y=1)
    for i, node in enumerate(range(23, 26)): pos[node] = (3 + i, -1)  # 支线 2：向下放置 (y=-1)
    for i, node in enumerate(range(26, 34)): pos[node] = (6 + i, -1)  # 支线 3：向下放置 (y=-1，完美错开支线2)

    plt.figure(figsize=(14, 6))
    
    # 绘制节点和配电线 (修改了 label)
    nx.draw_networkx_nodes(G, pos, node_color='#87CEFA', node_size=350, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color='black', width=2, label="配电线路 (常闭)")
    
    # 3. 分离联络线绘制（直线与弧线分离）
    straight_ties = [(8, 21), (12, 22), (18, 33)]
    curved_ties = [(9, 15), (25, 29)] 
    
    # 画不会重叠的直线联络线 (修改了 label)
    nx.draw_networkx_edges(G, pos, edgelist=straight_ties, edge_color='red', 
                           style='dashed', width=1.5, label="联络线 (常开)")
    
    # 画必须避开黑线的弧线联络线 
    nx.draw_networkx_edges(G, pos, edgelist=curved_ties, edge_color='red', 
                           style='dashed', width=1.5, 
                           connectionstyle="arc3,rad=-0.4", 
                           arrows=True, arrowstyle='-')

    # 图表细节美化 (修改了标题)
    plt.title("标准 IEEE 33 节点配电网拓扑图", fontsize=18, fontweight='bold')
    plt.axis("off")
    
    # 处理图例去重并设置图例字体大小
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_chinese_ieee33()