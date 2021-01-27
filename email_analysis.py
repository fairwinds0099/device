import csv

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pylab

import network
"""

the purpose of this class is to clean the email.csv by following these steps:
the csv file is results of the following SQL query:
    SELECT *  FROM `inside-291915.email.email_raw`
    where to_field like '%dtaa%'
    and from_field like '%dtaa%'
    LIMIT 16000
1. remove columns other than from, to and user
2. remove non dtaa emails.
3. extract new rows for multiple recipients.
4. generate a reference dict object holding names vs user  code
5. recreating a dataframe with coded users for to and user fields
6. calling centrality calcucations and writing them to an csv file 
"""

if __name__ == '__main__':
    df = pd.read_csv('//Users/apple4u/Downloads/bq-results-20201223-073702-gvno8a241ivr.csv')
    # dropping unused columns, only userid, from and to will be left
    df = df.drop(['id', 'date', 'content', 'cc_field', 'bcc_field', 'pc', 'size', 'attachments'], axis=1)
    print(df.head())
    users_unique = df.user.unique()
    print("the number unique users are: " + str(len(users_unique)))
    print("initial data size: " + str(len(df)))

    # creating a new dataframe which filters out non dtaa emails
    # multiple recipients are added as new rows so that new links can be formed.
    df_new = pd.DataFrame(columns=['useer', 'too'])
    for index, row in df.iterrows():
        to_field = str(row['to_field']).split(";")  # Ainsley.MacKensie.Doyle@dtaa.com etc.
        for i in range(len(to_field)):
            if to_field[i].__contains__('@dtaa.com'):  # filtering out non dtaa emails
                df_new = df_new.append({'useer': row['user'], 'froom': row['from_field'], 'too': to_field[i]},
                                       ignore_index=True)
    print('======= PRINTING DF WITH EXTRACTED EMAILS =========')
    print("extracted DF size: " + str(len(df_new)))
    print(df_new.head(10))

    # generating coded user vs from  dictionary to be used as reference
    froom_all = df_new.froom.values
    useer_all = df_new.useer.values
    dict_temp = {}
    for i in range(len(froom_all)):
        dict_temp[froom_all[i]] = useer_all[i]
    print("length of reference dict for user vs from fields is: " +  str(len(dict_temp)))

    # created empty final df with coded_to and user columns
    df_coded = pd.DataFrame(columns=['coded_to', 'user_common'])

    for index, row in df_new.iterrows():
        coded_too = ''
        for key in dict_temp.keys(): #iterating all keys to match the name to coded_user
            if row['too'] == key:
                coded_too = dict_temp.get(key)
                break
        df_coded = df_coded.append({'coded_to': coded_too, 'user_common': row['useer']}, ignore_index=True)

    print('======= PRINTING DF WITH CODED/TRANSFORMED EMAILS =========')
    print(df_coded.head(10))

    df_coded.to_csv(r'export_dataframe.csv', index=False, header=True) #50k records

    df_coded = df_coded.head(10000)
    G = nx.Graph()
    X = df_coded.user_common.values
    y = df_coded.coded_to.values

    print('======= GENERATING GRAPH =========')
    for i in range(len(df_coded)):
        G.add_node(X[i])
        G.add_node(y[i])
        G.add_edge(X[i], y[i])

    options = {
        "node_color": "skyblue",
        "edge_color": "#1017e0",
        "width": 1,
        # "edge_cmap": plt.cm.colors.TABLEAU_COLORS,
        #"with_labels": True,
        #"font_size": 8,
        "nodelist": G.nodes,
        #"node_size": 0.1
        "node_size": [G.degree[v] for v in G]
    }

    #nx.draw(G, pos, **options)
    nx.draw(G, pos=nx.spring_layout(G), **options)
    fig = pylab.matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)
    fig.savefig('email_fructerman.png', dpi=100)

#commented out for plotting purposes
    #print('======= CALCULATING GRAPH METRICS=========')
    #calculating 3 centrality metrics: degree. betweenness, closeness
    #network.calculate_centrality('pagerank', G)
