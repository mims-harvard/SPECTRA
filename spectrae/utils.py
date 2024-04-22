from networkx.algorithms.components import  connected_components

def is_integer(n):
    if isinstance(n, int):
        return True
    elif isinstance(n, float):
        return n.is_integer()
    else:
        return False
    
def is_clique(G):
    return G.size() == (G.order()*(G.order()-1))/2