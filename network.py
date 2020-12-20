import operator

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def calculate_centrality(centrality_type, graph):

    if centrality_type == 'degree':
        dict = nx.degree_centrality(graph)
    if centrality_type == 'betweenness':
        dict = nx.betweenness_centrality(graph)
    if centrality_type == 'closeness':
        dict = nx.closeness_centrality(graph)

    keymax = max(dict, key=dict.get)
    print(keymax)
    print(dict.get(keymax))

    sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
    print(sorted_x)
    print(len(dict))

    print(dict.get("DRR0162"))

    dfcentrality = pd.DataFrame.from_dict(sorted_x)
    dfcentrality.info()
    dfcentrality.head()
    dfcentrality.to_csv(r'/Users/apple4u/Desktop/goksel tez/_centrality_raw.csv', index=False)


if __name__ == '__main__':
    G = nx.Graph()
    peterson =  nx.petersen_graph()
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
