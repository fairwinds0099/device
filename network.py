import operator

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import csv


def calculate_centrality(centrality_type, graph):

    if centrality_type == 'degree':
        dict = nx.degree_centrality(graph)
    if centrality_type == 'betweenness':
        dict = nx.betweenness_centrality(graph)
    if centrality_type == 'closeness':
        dict = nx.closeness_centrality(graph)

    target_path = '/Users/apple4u/Desktop/goksel tez/' + centrality_type + '_centrality_raw.csv'
    with open(target_path, 'w') as f:
        w = csv.writer(f)
        w.writerows(dict.items())

#sample graph generating and plotting methods
if __name__ == '__main__':
    G = nx.Graph()
    peterson = nx.petersen_graph()
    df = pd.read_csv('/Users/apple4u/Downloads/logon_distinct.csv')
    df.info()
    df.head(5)
    X = df.user.values
    y = df.pc.values

    for i in range(len(df)):
        G.add_node(X[i])
        G.add_node(y[i])
        G.add_edge(X[i], y[i])

    nx.draw(G)
    plt.savefig('plot.png')

    #calculate_centrality('closeness', peterson)
