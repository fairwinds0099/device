import operator

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import csv

import pylab
from networkx.readwrite import json_graph
import json
from networkx.readwrite import json_graph


# noinspection PyShadowingBuiltins
def calculate_centrality(centrality_type, graph):
    if centrality_type == 'degree':
        dict = nx.degree_centrality(graph)
    if centrality_type == 'betweenness':
        dict = nx.betweenness_centrality(graph)
    if centrality_type == 'closeness':
        dict = nx.closeness_centrality(graph)
    if centrality_type == 'pagerank':
        dict = nx.pagerank(graph)
    target_path = '/Users/apple4u/Desktop/goksel tez/' + centrality_type + '_centrality_raw.csv'
    with open(target_path, 'w') as f:
        w = csv.writer(f)
        w.writerows(dict.items())


# sample graph generating and plotting methods
if __name__ == '__main__':
    # nx = nx.petersen_graph()
    G = nx.Graph()
    #df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/findings_data/logon_distinct_shrinked.csv')
    df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/findings_data/logon_distinct_shrinked.csv')

    df.info()

    X = df.user.values
    y = df.pc.values
    for i in range(len(df)):
        G.add_node(X[i])
        G.add_node(y[i])
        G.add_edge(X[i], y[i])

    #pos = nx.spring_layout(G, seed=63)  # Seed layout for reproducibility
    pos = nx.bipartite_layout(G, X);
    colors = range(20)
    options = {
        "node_color": "skyblue",
        R"edge_color": "#1017e0",
        "width": 0.2,
        #"edge_cmap": plt.cm.colors.TABLEAU_COLORS,
        "with_labels": True,
        "nodelist": G.nodes,
        "node_size": [G.degree[v] for v in G]
    }

    nx.draw(G, pos, **options)
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", edge_color="#1f78b4", pos=nx.bipartite_layout(G, X))
    #nx.draw(G, nodelist=G.nodes, node_size=[G.degree[v] for v in G], pos=nx.fruchterman_reingold_layout(G), **options)
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", pos=nx.shell_layout(G))
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", pos=nx.spring_layout(G))
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", pos=nx.kamada_kawai_layout(G))
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", edge_color="#1017e0", pos=nx.spectral_layout(G, 5))
    # returns empty
    # nx.draw_networkx(G, with_labels=False, node_size=0.1, width=0.1, node_color="skyblue", edge_color="#1017e0", pos=nx.planar_layout(G))
    # G is not planar
    fig = pylab.matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)
    fig.savefig('test2png.png', dpi=100)

    # calculate_centrality('algeabric_connectivity', G)
    # print(nx.algebraic_connectivity(G))

# with open('networkdata1.json', 'w') as outfile1:
#     outfile1.write(json.dumps(json_graph.node_link_data(G)))
