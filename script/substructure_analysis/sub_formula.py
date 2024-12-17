import pandas as pd 
import json
import re



if __name__ == "__main__":
    cyp3a4_subs = pd.read_csv('../../outputs/substructure_analysis/functional_subs/cyp3a4_negative_attri_df.csv')
    cyp2c19_subs = pd.read_csv('../../outputs/substructure_analysis/functional_subs/cyp2c19_negative_attri_df.csv')

    subs_merge = cyp3a4_subs['sub_smi'].tolist() + cyp2c19_subs['sub_smi'].tolist()
    cyp_nega_attri_subs = list(set(subs_merge))
    cyp_nega_attri_subs.sort()
    cyp_nega_attri_subs = [re.sub('\[[0-9]+\*\]', '*', smi) for smi in cyp_nega_attri_subs]

    file_path = '../../outputs/substructure_analysis/functional_subs/cyp_nega_subs_for_rxn_tpl.json'
    with open(file_path, 'w') as f:
        json.dump(cyp_nega_attri_subs, f)


    