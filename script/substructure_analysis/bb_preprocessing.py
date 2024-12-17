import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json

if __name__ == "__main__":

    ''''
    Principles of building block preprocessing: The substructures are tox alerts(cyp alerts), 
    but not reaction intermediates or common molecular scaffolds (which would result in the loss of a large number of molecules). 
    The file tox_posi_subs_for_rxn_tpl.json needs to be manually managed.
    '''

    bblock = pd.read_csv('../../data/molecular_optimization/assets/building-blocks/enamine-us-smiles.csv.gz')
    bblock_ls = list(set(bblock['SMILES'].tolist()))
    bblock_ls.sort()                    

    sub_file_path = '../../outputs/substructure_analysis/functional_subs/tox_posi_subs_for_rxn_tpl.json'
    with open(sub_file_path, 'r') as f:
        tox_sub_ls = json.load(f)      
    tox_sub_mols = [Chem.MolFromSmarts(sub) for sub in tox_sub_ls]     

    bblock_dict = {}
    for smi in tqdm(bblock_ls):             
        mol = Chem.MolFromSmiles(smi)
        for tox_sub_mol in tox_sub_mols:
            if mol.HasSubstructMatch(tox_sub_mol):
                bblock_dict[smi] = False
                break
            else:
                bblock_dict[smi] = True    

    assert len(bblock_dict.keys()) == len(bblock_ls)
    assert len(bblock_dict.values()) == len(bblock_ls)

    filtered_bblock_df = pd.DataFrame.from_dict(bblock_dict, orient='index', columns=['select'])
    filtered_bblock_df.insert(0, 'smiles', filtered_bblock_df.index)
    print(len(filtered_bblock_df))      

    select_bblock_df = filtered_bblock_df.loc[filtered_bblock_df['select'] == True]
    print(len(select_bblock_df))        
    select_bblock_df.to_csv('../../data/molecular_optimization/assets/building-blocks/preprocessed_building_blocks.csv', index=False)