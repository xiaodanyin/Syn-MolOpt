import rdkit.Chem as Chem
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
import pandas as pd
from tqdm import tqdm

class NotCanonicalizableSmilesException(ValueError):
    pass

def canonicalize_smi(smi, remove_atom_mapping=False):
    r"""
    Canonicalize SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise NotCanonicalizableSmilesException("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)

def get_canonical_precursors(rxn):
    reactants, products = rxn.split(">>")
    reactants = reactants.split('.')
    reactants = [x for x in reactants if ':' in x]
    reactants = '.'.join(reactants)
    try:
        precursors = canonicalize_smi(reactants, True)
    except NotCanonicalizableSmilesException:
        return ""

    return f'{reactants}>>{products}', precursors

def get_templates(rxn_smi, prec, no_special_groups, radius, add_brackets=True):
    """
    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param prec: Canonical smiles string of precursor
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates
    :param add_brackets: Whether to add brackets to make template pseudo-unimolecular

    :return: Template
    """    
    #Extract:
    try:
        rxn_split = rxn_smi.split(">")
        reaction={"_id":0,"reactants":rxn_split[0],"spectator":rxn_split[1],"products":rxn_split[2]}
        template = extract_from_reaction(reaction, no_special_groups=no_special_groups, radius=radius)["reaction_smarts"]
        # template = extract_from_reaction(reaction)["reaction_smarts"]
        if add_brackets:
            template = "(" + template.replace(">>", ")>>")
    except:
        template = None  
    #Validate:
    if template != None:
        rct = rdchiralReactants(rxn_smi.split(">")[-1])
        try:
            rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except:
            outcomes =[]
        if not prec in outcomes:
            template=None
    return template

def get_templates_temprel(reaction, no_special_groups, radius):
    """
    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates

    :return: Template
    """    
    try:
        return extract_from_reaction(reaction,no_special_groups=no_special_groups,radius=radius)
    except Exception as e:
        return {
            'reaction_id': reaction['_id'],
            'error': str(e)
        }

def switch_direction(template, brackets=True):
    """Computes reversed templates.

    :param template: Reaction template
    :param brackets: Boolean whether template contains brackets to make the right side unimolecular.

    :return: Reversed template
    """
    if brackets:
        left_side=template.split(">")[0][1:-1]
        right_side=template.split(">")[-1]
        reverse_template="("+right_side+")>>"+left_side
    else:
        left_side=template.split(">")[0]
        right_side=template.split(">")[-1]
        reverse_template=right_side+">>"+left_side
    return reverse_template

if __name__ == "__main__":

    uspto_1976 = pd.read_csv('../../data/functional_rxn_tpl/uspto_reaction/1976_Sep2016_USPTOgrants_smiles_mapped.tsv', delimiter='\t')
    uspto_2001 = pd.read_csv('../../data/functional_rxn_tpl/uspto_reaction/2001_Sep2016_USPTOapplications_smiles_mapped.tsv', delimiter='\t')
    uspto_rxn_merge = pd.concat([uspto_1976, uspto_2001], axis=0)
    uspto_rxn = uspto_rxn_merge.drop_duplicates(subset=['mapped_rxn']).reset_index(drop=True)

    def get_templates_with_timeout(rxn_smi, prec, no_special_groups, radius, add_brackets=True):
        try:
            return get_templates(rxn_smi, prec, no_special_groups, radius, add_brackets=add_brackets)
        except Exception as e:
            print(e)
            return ''

    rxn_template = []
    for rxn in tqdm(uspto_rxn['mapped_rxn'].tolist()):
        rxn_, prec = get_canonical_precursors(rxn)
        template = get_templates_with_timeout(rxn_, prec, True, 1)
        rxn_template.append(template)
    uspto_rxn['retro_templates'] = rxn_template
    uspto_rxn = uspto_rxn[uspto_rxn['retro_templates'].notna()]

    uspto_rxn.to_csv('../../data/functional_rxn_tpl/retro_templates/uspto_reaction_retro_templates.csv', index=False)