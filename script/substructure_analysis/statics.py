import pandas as pd
from rdkit import Chem
import statistics


def init_statics(task):
    cyp_sub = pd.read_csv(f'../../outputs/substructure_analysis/attribution/{task}_brics_frag.csv')
    set_frag_smis = list(set(cyp_sub['frag_smiles'].tolist()))
    cyp_atom_mols = [Chem.MolFromSmiles(x) for x in cyp_sub['frag_smiles'].tolist()]
    cyp_atom_nums = [mol.GetNumAtoms() for mol in cyp_atom_mols]
    cyp_sub['sub_atom_num'] = cyp_atom_nums

    average_attri_dict = {
        'sub_smi': [],
        "atom_num": [],
        "count": [],
        'attri_average': [],
    }    
    for frag_smi in set_frag_smis:
        cyp_data = cyp_sub.loc[cyp_sub['frag_smiles'] == frag_smi]
        atom_num = cyp_data['sub_atom_num'].tolist()[0]
        cnt = len(cyp_data)
        average_attri = statistics.mean(cyp_data['attribution'].tolist())
        average_attri_dict['sub_smi'].append(frag_smi)
        average_attri_dict['atom_num'].append(atom_num)
        average_attri_dict['count'].append(cnt)
        average_attri_dict['attri_average'].append(average_attri)

    assert len(average_attri_dict['sub_smi']) == len(set_frag_smis)
    average_attri_df = pd.DataFrame.from_dict(average_attri_dict, orient='columns')
    sorted_average_attri_df = average_attri_df.sort_values(by='attri_average', ascending=False)

    return sorted_average_attri_df

if __name__ == "__main__":

    task = 'cyp2c19' 
    average_attri_df = init_statics(task)
    nega_attri_df = average_attri_df.loc[average_attri_df['attri_average'] < -0.2]

    cnt_nega_attri_df = nega_attri_df.loc[nega_attri_df['count'] >= 2]
    num_nega_attri_df = cnt_nega_attri_df.loc[cnt_nega_attri_df['atom_num'] >= 2] 
    selected_nega_attri_df = num_nega_attri_df.loc[cnt_nega_attri_df['atom_num'] < 12] 

    print(selected_nega_attri_df.shape)
    selected_nega_attri_df.to_csv(f'../../outputs/substructure_analysis/functional_subs/{task}_negative_attri_df.csv', index=False)

