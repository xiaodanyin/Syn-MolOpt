from typing import Callable, Tuple, List, Dict
import numpy as np
import pytorch_lightning as pl
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import BallTree
import random
from synnet.encoding.distances import cosine_distance
from synnet.encoding.fingerprints import mol_fp
from synnet.encoding.utils import one_hot_encoder
from synnet.utils.data_utils import Reaction, SyntheticTree
from rdkit import RDLogger
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)


def count_heavy_atoms(molecule):
    return len([atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1])  

def can_react(state, rxns: List[Reaction]) -> Tuple[int, List[bool]]:
    """
    Determines if two molecules can react using any of the input reactions.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Returns:
        np.ndarray: The sum of the reaction mask tells us how many reactions are
             viable for the two molecules.
        np.ndarray: The reaction mask, which masks out reactions which are not
            viable for the two molecules.
    """
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction((mol1, mol2)) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask

def get_action_mask(state: list, rxns: List[Reaction]) -> np.ndarray:
    """
    Determines which actions can apply to a given state in the synthetic tree
    and returns a mask for which actions can apply.

    Args:
        state (np.ndarray): The current state in the synthetic tree.
        rxns (list of Reaction objects): Contains available reaction templates.

    Raises:
        ValueError: There is an issue with the input state.

    Returns:
        np.ndarray: The action mask. Masks out unviable actions from the current
            state using 0s, with 1s at the positions corresponding to viable
            actions.
    """
    # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
    if len(state) == 0:
        mask = [1, 0, 0, 0]
    elif len(state) == 1:
        mask = [1, 1, 0, 1]
    elif len(state) == 2:
        can_react_, _ = can_react(state, rxns)
        if can_react_:
            mask = [0, 1, 1, 0]
        else:
            mask = [0, 1, 0, 0]
    else:
        raise ValueError("Problem with state.")
    return np.asarray(mask, dtype=bool)

def get_reaction_mask(smi: str, rxns: List[Reaction]):
    """
    Determines which reaction templates can apply to the input molecule.

    Args:
        smi (str): The SMILES string corresponding to the molecule in question.
        rxns (list of Reaction objects): Contains available reaction templates.

    Raises:
        ValueError: There is an issue with the reactants in the reaction.

    Returns:
        reaction_mask (list of ints, or None): The reaction template mask. Masks
            out reaction templates which are not viable for the input molecule.
            If there are no viable reaction templates identified, is simply None.
        available_list (list of lists, or None): Contains available reactants if
            at least one viable reaction template is identified. Else is simply
            None.
    """
    reaction_mask = [int(rxn.is_reactant(smi)) for rxn in rxns]

    if sum(reaction_mask) == 0:
        return None, None

    available_list = []
    mol = rdkit.Chem.MolFromSmiles(smi)
    for i, rxn in enumerate(rxns):
        if reaction_mask[i] and rxn.num_reactant == 2:

            if rxn.is_reactant_first(mol):
                available_list.append(rxn.available_reactants[1])
            elif rxn.is_reactant_second(mol):
                available_list.append(rxn.available_reactants[0])
            else:
                raise ValueError("Check the reactants")

            if len(available_list[-1]) == 0:
                reaction_mask[i] = 0

        else:
            available_list.append([])

    return reaction_mask, available_list

def nn_search(
    _e: np.ndarray, _tree: BallTree, _k: int = 1
) -> Tuple[float, float]:  
    """
    Conducts a nearest neighbor search to find the molecule from the tree most
    simimilar to the input embedding.

    Args:
        _e (np.ndarray): A specific point in the dataset.
        _tree (sklearn.neighbors._kd_tree.KDTree, optional): A k-d tree.
        _k (int, optional): Indicates how many nearest neighbors to get.
            Defaults to 1.

    Returns:
        float: The distance to the nearest neighbor.
        int: The indices of the nearest neighbor.
    """
    dist, ind = _tree.query(_e, k=_k)
    return dist[0][0], ind[0][0]

def nn_search_rt1(_e: np.ndarray, _tree: BallTree, _k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    dist, ind = _tree.query(_e, k=_k)
    return dist[0], ind[0]

def set_embedding(
    z_target: np.ndarray, state: List[str], nbits: int, _mol_embedding: Callable
) -> np.ndarray:
    """
    Computes embeddings for all molecules in the input space.
    Embedding = [z_mol1, z_mol2, z_target]

    Args:
        z_target (np.ndarray): Molecular embedding of the target molecule.
        state (list): State of the synthetic tree, i.e. list of root molecules.
        nbits (int): Length of fingerprint.
        _mol_embedding (Callable): Computes the embeddings of molecules in the state.

    Returns:
        embedding (np.ndarray): shape (1,d+2*nbits)
    """
    z_target = np.atleast_2d(z_target)  
    if len(state) == 0:
        z_mol1 = np.zeros((1, nbits))
        z_mol2 = np.zeros((1, nbits))
    elif len(state) == 1:
        z_mol1 = np.atleast_2d(_mol_embedding(state[0]))
        z_mol2 = np.zeros((1, nbits))
    elif len(state) == 2:
        z_mol1 = np.atleast_2d(_mol_embedding(state[0]))
        z_mol2 = np.atleast_2d(_mol_embedding(state[1]))
    else:
        raise ValueError
    embedding = np.concatenate([z_mol1, z_mol2, z_target], axis=1)
    return embedding  

def syn_molopt_synthetic_tree_decoder(
    z_target: np.ndarray,
    building_blocks: List[str],
    bb_dict: Dict[str, int],
    reaction_templates: List[Reaction],
    functional_templates: List[Reaction],
    mol_embedder,
    action_net: pl.LightningModule,
    reactant1_net: pl.LightningModule,
    rxn_net: pl.LightningModule,
    reactant2_net: pl.LightningModule,
    bb_emb: np.ndarray,
    n_bits: int,
    max_step: int = 15,
    k_reactant1: int = 1,
) -> Tuple[SyntheticTree, int]:
    """
    Computes a synthetic tree given an input molecule embedding.
    Uses the Action, Reaction, Reactant1, and Reactant2 networks and a greedy search.

    Args:
        z_target (np.ndarray): Embedding for the target molecule
        building_blocks (list of str): Contains available building blocks
        bb_dict (dict): Building block dictionary
        reaction_templates (list of Reactions): Contains 91 publicly available reaction templates
        functional_templates (list of Reactions): Contains functional reaction templates
        mol_embedder: GNN to use for obtaining molecular embeddings
        action_net: The action network
        reactant1_net: The reactant1 network
        rxn_net: The reaction network
        reactant2_net: The reactant2 network
        bb_emb (list): Contains purchasable building block embeddings.
        n_bits (int): Length of fingerprint.
        max_step (int, optional): Maximum number of steps to include in the
            synthetic tree

    Returns:
        tree (SyntheticTree): The final synthetic tree.
        act (int): The final action (to know if the tree was "properly"
            terminated).
    """

    tree = SyntheticTree()
    mol_recent = None
    kdtree = mol_embedder  
    
    for i in range(max_step):
        print(i)
    
        state = tree.get_state()  
        z_state = set_embedding(z_target, state, nbits=n_bits, _mol_embedding=mol_fp)

        # Predict reaction action
        action_proba = action_net(torch.Tensor(z_state)) 
        action_proba = action_proba.squeeze().detach().numpy() + 1e-10
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)
        if act == 3:  
            break

        # Predict 1st molecule
        z_mol1 = reactant1_net(torch.Tensor(z_state))   
        z_mol1 = z_mol1.detach().numpy()  
        if act == 0:
            k = k_reactant1 if mol_recent is None else 1
            _, idxs = kdtree.query(z_mol1, k=k)  
            mol1 = building_blocks[idxs[0][k - 1]]
        elif act == 1 or act == 2:
            mol1 = mol_recent
        else:
            raise ValueError(f"Unexpected action {act}.")
        z_mol1 = mol_fp(mol1)
        z_mol1 = np.atleast_2d(z_mol1) 

        # Predict rxn template
        z = np.concatenate([z_state, z_mol1], axis=1)
        reaction_proba = rxn_net(torch.Tensor(z))   
        reaction_proba = reaction_proba.squeeze().detach().numpy() + 1e-10  
        reaction_proba = reaction_proba[:91] 
        if act == 0 or act == 1:  
            reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
        else:  
            _, reaction_mask = can_react(tree.get_state(), reaction_templates)
            available_list = [[] for rxn in reaction_templates]  
        if sum(reaction_mask) == 0:
            if len(state) == 1:  
                act = 3
                break
            else:
                break  
        rxn_id = np.argmax(reaction_proba * reaction_mask)
        rxn = reaction_templates[rxn_id]

        # Predict 2nd reactant
        if rxn.num_reactant == 2:
            if act == 2:  
                temp = set(state) - set([mol1])
                mol2 = temp.pop()
            else: 
                x_rxn = one_hot_encoder(rxn_id, len(reaction_templates) + len(functional_templates))
                x_rct2 = np.concatenate([z_state, z_mol1, x_rxn], axis=1)
                z_mol2 = reactant2_net(torch.Tensor(x_rct2))    
                z_mol2 = z_mol2.detach().numpy()
                available = available_list[rxn_id] 
                available = [bb_dict[available[i]] for i in range(len(available))]  
                temp_emb = bb_emb[available]
                available_tree = BallTree(temp_emb, metric=cosine_distance)  
                dist, ind = nn_search(z_mol2, _tree=available_tree)
                mol2 = building_blocks[available[ind]]
        else:
            mol2 = None

        # Syntree updates
        mol_product = rxn.run_reaction((mol1, mol2))
        if mol_product is None or Chem.MolFromSmiles(mol_product) is None:
            if len(state) == 1:  
                act = 3
                break
            else:
                break
        tree.update(act, int(rxn_id), mol1, mol2, mol_product)
        mol_recent = mol_product

        # Functional processing
        func_tpl_mask = []
        mol_product_mol = Chem.MolFromSmiles(mol_product)
        heavy_atoms_mol_product = count_heavy_atoms(mol_product_mol)

        for func_tpl in functional_templates:
            reaction = AllChem.ReactionFromSmarts(func_tpl.smirks)
            reaction.Initialize()
            is_reactant = reaction.IsMoleculeReactant(mol_product_mol)
            func_tpl_mask.append(int(is_reactant))

        if sum(func_tpl_mask) != 0:
            non_zero_indices = [i for (i, element) in enumerate(func_tpl_mask) if element != 0]
            if len(non_zero_indices) != 0:
                func_tpl_id = random.choice(non_zero_indices)
                func_tpl_used = functional_templates[func_tpl_id]
                rxn_tpl = func_tpl_used.smirks
                func_tpl_used = AllChem.ReactionFromSmarts(rxn_tpl)
                func_product_mol = func_tpl_used.RunReactants([mol_product_mol])
                func_product = Chem.MolToSmiles(func_product_mol[0][0])

                heavy_atoms_func_product = count_heavy_atoms(func_product_mol[0][0])
                difference = abs(heavy_atoms_mol_product - heavy_atoms_func_product)
                if difference <= 6:            
                    func_tpl_id_all = func_tpl_id + len(reaction_templates)
                    act = 1
                    tree.update(act, int(func_tpl_id_all), mol_product, None, func_product)
                    mol_recent = func_product
                else:
                    pass
            else:
                pass
        else:
            pass

    if act != 3:
        tree = tree
    else:
        tree.update(act, None, None, None, None)

    return tree, act
