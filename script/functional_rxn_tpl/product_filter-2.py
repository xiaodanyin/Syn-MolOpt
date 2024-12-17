import pandas as pd
import json
from rdkit import Chem
from tqdm import tqdm


if __name__ == "__main__":

    sub_file_path = '../../outputs/substructure_analysis/functional_subs/tox_nega_subs_for_rxn_tpl.json'
    with open(sub_file_path, 'r') as f:
        sub_ls = json.load(f)
    sub_mols = [Chem.MolFromSmiles(sub) for sub in sub_ls] 

    template_pdt_without_tox_subs = pd.read_csv('../../data/functional_rxn_tpl/retro_templates/retro_templates_product_without_tox_subs.csv')

    select_idx = []
    for i, rxn_smi in enumerate(tqdm(template_pdt_without_tox_subs['retro_templates'].tolist())):
        template_pdt = rxn_smi.split('>>')[0]
        template_pdt_ = template_pdt[1:-1]
        pdt_mol = Chem.MolFromSmarts(template_pdt_)

        for sub_mol in sub_mols:
            if sub_mol.HasSubstructMatch(pdt_mol):
                match_pdt_idx = sub_mol.GetSubstructMatch(pdt_mol)
                match_pdt_idx = sorted(list(match_pdt_idx))
                match_detoxic_sub_idx = [a.GetIdx() for a in sub_mol.GetAtoms() if a.GetSymbol() != "*"] 
                if match_pdt_idx ==  match_detoxic_sub_idx:
                    select_idx.append(i)
                    break
    
    templates_pdt_with_detox_subs = template_pdt_without_tox_subs.loc[select_idx]    

    templates = []
    for retro_template in templates_pdt_with_detox_subs['retro_templates'].tolist():
        template_rct = retro_template.split('>>')[-1]
        template_pdt = retro_template.split('>>')[0]
        template = f'{template_rct}>>{template_pdt}'
        templates.append(template)
    templates_pdt_with_detox_subs['templates'] = templates
    
    # !!!!!!!!! functional reaction templates require manual management !!!!!!!!!
    templates_pdt_with_detox_subs.to_csv('../../outputs/functional_rxn_tpl/tox_functional_reaction_templates.csv', index = False)   

    

    




