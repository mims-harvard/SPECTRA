import pickle
import networkx as nx
from equibind_similarity_util import is_similar
import pandas as pd
from tqdm import tqdm

edges = pickle.load(open('equibind_edges', 'rb'))
G = nx.Graph()
df = pd.read_csv('equibind_dataset.csv')

for edge in tqdm(edges):
    #We redo calculation here we will only consider two sequence as similar if regardless of order they are similar
    #This is post-processsing so I don't have to redo the entire calculation

    seq_1=df.iloc[edge[0]].seq
    seq_2=df.iloc[edge[1]].seq
    smile_1 = df.iloc[edge[0]].smiles
    smile_2 = df.iloc[edge[1]].smiles

    if is_similar(seq_1, seq_2, smile_1, smile_2) and is_similar(seq_2, seq_1, smile_2, smile_1):
        G.add_edge(edge[0], edge[1])

print(f"Have {len(G.nodes())} nodes and {len(G.edges())} edges and {nx.number_connected_components(G)} connected components")
import pickle
pickle.dump(G, open('equibind_graph_v2.pickle', 'wb'))
            

