import pandas as pd
import json
from rdkit import Chem
from tqdm import tqdm


if __name__ == "__main__":

    sub_file_path = '../../outputs/substructure_analysis/functional_subs/tox_posi_subs_for_rxn_tpl.json'
    with open(sub_file_path, 'r') as f:
        sub_ls = json.load(f)      
    sub_mols = [Chem.MolFromSmiles(sub) for sub in sub_ls] 
    uspto_retro_templates = pd.read_csv('../../data/functional_rxn_tpl/retro_templates/uspto_reaction_retro_templates.csv')    

    select_idx = []
    for i, rxn_smi in enumerate(tqdm(uspto_retro_templates['retro_templates'].tolist())):
        if not pd.isna(rxn_smi):
            template_reactants = rxn_smi.split('>>')[-1] 
            reactant_mols = [Chem.MolFromSmarts(template_reactant) for template_reactant in template_reactants.split('.')]

            inner_select_idx = []
            for toxic_sub_mol in sub_mols:
                if len(inner_select_idx) > 0:
                    break
                else:
                    for template_mol in reactant_mols:
                        if toxic_sub_mol.HasSubstructMatch(template_mol):
                            match_template_idx = toxic_sub_mol.GetSubstructMatch(template_mol)
                            match_template_idx = sorted(list(match_template_idx))
                            match_toxic_sub_idx = [a.GetIdx() for a in toxic_sub_mol.GetAtoms() if a.GetSymbol() != "*"]
                            
                            if match_template_idx == match_toxic_sub_idx:
                                select_idx.append(i)
                                inner_select_idx.append(i)
                                break

    uspto_rxn_contain_tox_subs = uspto_retro_templates.loc[select_idx]
    uspto_rxn_contain_tox_subs.to_csv('../../data/functional_rxn_tpl/retro_templates/retro_templates_reactant_with_tox_subs.csv', index=False)