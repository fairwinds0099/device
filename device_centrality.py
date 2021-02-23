import networkx as nx
import pandas as pd
if __name__=='__main__':
    G = nx.Graph()
    df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/findings_data/logon_distinct.csv')

    df.info()
    X = df.user.values
    y = df.pc.values
    for i in range(len(df)):
        G.add_node(X[i])
        G.add_node(y[i])
        G.add_edge(X[i], y[i])

    import network
    network.calculate_centrality('common_neighbor', G)
