import pandas as pd
import json
from rdkit import Chem
from tqdm import tqdm

if __name__ == "__main__":

    sub_file_path = '../../outputs/substructure_analysis/functional_subs/tox_posi_subs_for_rxn_tpl.json'
    with open(sub_file_path, 'r') as f:
        sub_ls = json.load(f)      
    sub_mols = [Chem.MolFromSmiles(sub) for sub in sub_ls] 

    template_rct_with_tox_subs = pd.read_csv('../../data/functional_rxn_tpl/retro_templates/retro_templates_reactant_with_tox_subs.csv') 
    template_rct_with_tox_subs = template_rct_with_tox_subs.drop_duplicates(subset=['retro_templates']).reset_index(drop=True)   

    drop_idx = []
    for i, rxn_smi in enumerate(template_rct_with_tox_subs['retro_templates'].tolist()):
        if not pd.isna(rxn_smi):
            template_pdt = rxn_smi.split('>>')[0]
            template_pdt_ = template_pdt[1:-1]
            pdt_mol = Chem.MolFromSmarts(template_pdt_)

            inner_drop_idx = []
            for sub_mol in sub_mols:
                if len(inner_drop_idx) > 0:
                    break
                else:
                    if sub_mol.HasSubstructMatch(pdt_mol):
                        match_pdt_idx = sub_mol.GetSubstructMatch(pdt_mol)
                        match_pdt_idx = sorted(list(match_pdt_idx))
                        match_toxic_sub_idx = [a.GetIdx() for a in sub_mol.GetAtoms() if a.GetSymbol() != "*"]

                        if match_pdt_idx == match_toxic_sub_idx:
                            drop_idx.append(i)
                            inner_drop_idx.append(i)
                            break

    template_pdt_without_tox_subs = template_rct_with_tox_subs.drop(drop_idx)
    template_pdt_without_tox_subs.to_csv('../../data/functional_rxn_tpl/retro_templates/retro_templates_product_without_tox_subs.csv', index=False)