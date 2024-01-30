import pandas as pd
from equibind_similarity_util import is_similar
from tqdm import tqdm

posebusters_set = pd.read_csv('Posebusters_test_set.csv')
equibind_set = pd.read_csv('equibind_dataset.csv')
equibind_train = [i.rstrip() for i in open('EquiBind_train', 'r').readlines()]
equibind_test = [i.rstrip() for i in open('EquiBind_test', 'r').readlines()]
equibind_train_index = equibind_set[equibind_set['id'].isin(equibind_train)]
equibind_test_index = equibind_set[equibind_set['id'].isin(equibind_test)]
posebusters_set = posebusters_set.rename(columns={"protein": "seq"})
astex_diverse_set = pd.read_csv('astex_diverse_set.csv')
astex_diverse_set = astex_diverse_set.rename(columns={"protein": "seq"})
LP_PDBind = pd.read_csv('LP_PDBBind.csv')
LP_PDBind = LP_PDBind.dropna()

results = {}

def calc_prop_similarity(train, test, results):
    num_similar = 0
    
    
    for seq_1, smile_1 in test:
        similar = False
        for seq_2, smile_2 in train:
            if is_similar(seq_1, seq_2, smile_1, smile_2) and is_similar(seq_2, seq_1, smile_2, smile_1):
                num_similar += 1
                results[f'{seq_1}-{smile_1}'] = f'{seq_2}-{smile_2}'
                similar = True
                break
        if not similar:
            results[f'{seq_1}-{smile_1}'] = 'None'

    return num_similar, len(test), results

if __name__ == '__main__':
    import sys
    dataset_to_test = int(sys.argv[1])
    if dataset_to_test == 0:
        to_compare = 'astex'
    elif dataset_to_test == 1:
        to_compare = 'posebusters'
    elif dataset_to_test == 2:
        to_compare = 'equibind'
    elif dataset_to_test == 3:
        to_compare = 'LP'

    train = equibind_train_index[['seq', 'smiles']].values.tolist()

    if to_compare == 'astex':
        test = astex_diverse_set[['seq', 'smiles']].values.tolist()
    elif to_compare == 'posebusters':
        test = posebusters_set[['seq', 'smiles']].values.tolist()
    elif to_compare == 'equibind':
        test = equibind_test_index[['seq', 'smiles']].values.tolist()
    elif to_compare == 'LP':
        train = LP_PDBind[LP_PDBind['new_split'] == 'train'][['seq', 'smiles']].values.tolist()
        test= LP_PDBind[LP_PDBind['new_split'] == 'test'][['seq', 'smiles']].values.tolist()

    num_similar, total, results = calc_prop_similarity(train, test, results)

    import pickle
    pickle.dump(results, open(f'equibind_similarity_calculations/similarity_results_{to_compare}_equibind', 'wb'))

    result_file = open(f"equibind_similarity_calculations/{to_compare}_result.txt", "w")
    result_file.write(f"{num_similar}\t{total}\n")
    result_file.close()

    print(num_similar)
    print(total)

