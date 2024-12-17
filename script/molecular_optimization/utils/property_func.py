import pandas as pd
from dgl import DGLGraph
from rdkit.Chem import MolFromSmiles
import torch as th
from rdkit import Chem
import numpy as np
import os
from dgl.data.graph_serialize import save_graphs
import json
from torch import nn
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
import torch.nn.functional as F
import random
import dgl
from torch.utils.data import DataLoader
import pickle


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index

def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)

def construct_RGCN_mol_graph_from_smiles(smiles, smask):
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    smask_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom)
        atoms_feature_all.append(atom_feature)
        if i in smask:
            smask_list.append(0)
        else:
            smask_list.append(1)
    g.ndata["node"] = th.tensor(atoms_feature_all)
    g.ndata["smask"] = th.tensor(smask_list).float()

    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    g.edata["edge"] = th.tensor(etype_feature_all)
    return g

def build_mol_graph_data(dataset_smiles):
    dataset_gnn = []
    failed_molecule = []
    labels = [0] * len(dataset_smiles)
    split_index = [0] * len(dataset_smiles)
    smilesList = dataset_smiles
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        try:
            g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=[])
            molecule = [smiles, g_rgcn, labels[i], split_index[i]]
            dataset_gnn.append(molecule)
        except:
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    return dataset_gnn

def _load_config(prop_name):
    config_path = os.path.join('utils', './property_config.json')
    with open(config_path, "r") as f:
        tox_config = json.load(f)
        return tox_config[prop_name]



class RGCNLayer(nn.Module):
    """Single layer RGCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    num_rels: int
        Number of bond type
    activation : activation function
        Default to be ReLU
    loop: bool:
        Whether to use self loop
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    rgcn_drop_out : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    """
    
    def __init__(self, in_feats, out_feats, num_rels=65, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()
        
        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                             num_bases=None, bias=True, activation=activation,
                                             self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
        
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
    
    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: th.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        th.cuda.empty_cache()
        return new_feats

class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """
    
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )
    
    def forward(self, g, feats, smask):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        smask: substructure mask, atom node for 0, substructure node for 1.

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            weight = self.atom_weighting(g.ndata['h']) * smask
            g.ndata['w'] = weight
            h_g_sum = sum_nodes(g, 'h', 'w')
        return h_g_sum, weight



class BaseGNN(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    
    def __init__(self, gnn_rgcn_out_feats, ffn_hidden_feats, ffn_dropout=0.25, classification=True):
        super(BaseGNN, self).__init__()
        self.classification = classification
        self.rgcn_gnn_layers = nn.ModuleList()
        self.readout = WeightAndSum(gnn_rgcn_out_feats)
        self.fc_layers1 = self.fc_layer(ffn_dropout, gnn_rgcn_out_feats, ffn_hidden_feats)
        self.fc_layers2 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.fc_layers3 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.predict = self.output_layer(ffn_hidden_feats, 1)
    
    def forward(self, rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats):
        """Multi-task prediction for a batch of molecules
        """
        # Update atom features with GNNs
        for rgcn_gnn in self.rgcn_gnn_layers:
            rgcn_node_feats = rgcn_gnn(rgcn_bg, rgcn_node_feats, rgcn_edge_feats)
        # Compute molecule features from atom features and bond features
        graph_feats, weight = self.readout(rgcn_bg, rgcn_node_feats, smask_feats)
        h1 = self.fc_layers1(graph_feats)
        h2 = self.fc_layers2(h1)
        h3 = self.fc_layers3(h2)
        out = self.predict(h3)
        return out, weight
    
    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class RGCN(BaseGNN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    gnn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    
    def __init__(self, ffn_hidden_feats, rgcn_node_feats, rgcn_hidden_feats, rgcn_drop_out=0.25, ffn_dropout=0.25,
                 classification=True):
        super(RGCN, self).__init__(gnn_rgcn_out_feats=rgcn_hidden_feats[-1],
                                       ffn_hidden_feats=ffn_hidden_feats,
                                       ffn_dropout=ffn_dropout,
                                       classification=classification,
                                       )
        for i in range(len(rgcn_hidden_feats)):
            rgcn_out_feats = rgcn_hidden_feats[i]
            self.rgcn_gnn_layers.append(RGCNLayer(rgcn_node_feats, rgcn_out_feats, loop=True,
                                                  rgcn_drop_out=rgcn_drop_out))
            rgcn_node_feats = rgcn_out_feats


def load_model(config):
    model_info_path = os.path.join(config['model_info'])
    model_info_dict = th.load(model_info_path, map_location=th.device('cpu'))
    task_flag = config['task'].split('-')[0]

    with open(f'/mnt/d/work/Syn-MolOpt/outputs/substructure_analysis/result/hyperparameter_{task_flag}.pkl', 'rb') as f:
        hyperparameter = pickle.load(f)

    model = RGCN(
        ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
        ffn_dropout=hyperparameter['ffn_drop_out'],
        rgcn_node_feats=hyperparameter['in_feats'],
        rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
        rgcn_drop_out=hyperparameter['rgcn_drop_out'],
        classification=hyperparameter['classification']
    )

    model_state = model_info_dict['model_state_dict']
    model.load_state_dict(model_state)
    
    return model, config

def built_mol_graph_data_and_save(origin_data,):
    data_set_gnn = build_mol_graph_data(dataset_smiles=origin_data)
    smiles, g_rgcn, labels, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': th.tensor(labels)}
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    return split_index_pd, g_rgcn, graph_labels


def collate_molgraphs(data):
    smiles, g_rgcn, labels, smask, sub_name = map(list, zip(*data))
    rgcn_bg = dgl.batch(g_rgcn)
    labels = th.tensor(labels)
    return smiles, rgcn_bg, labels, smask, sub_name

class RGCN_wraper:
    def __init__(self, model, config) -> None:
        self.config = config
        self.model = model

    def predict(self, smiles=None, device=th.device('cpu')):
        dummy_smiles = ['CCCC']
        # smiles_ls = dummy_smiles + smiles
        dummy_smiles.append(smiles)
        assert dummy_smiles

        split_index_pd, g_rgcn, graph_labels = built_mol_graph_data_and_save(
        origin_data=dummy_smiles,         
        )

        dataset = load_graph_from_csv_bin_for_splited(
        homog=g_rgcn,
        detailed_information=graph_labels,
        group_data=split_index_pd,
        classification=True,
        seed=2022,
        random_shuffle=False
    )
        # print("Molecule graph is loaded!")

        data_loader = DataLoader(dataset=dataset,
                                batch_size=128,
                                collate_fn=collate_molgraphs)
        
        self.model = self.model.to(device)
        self.model.eval()
        with th.no_grad():
            for batch_data in data_loader:
                smiles, rgcn_bg, labels, smask_idx, sub_name = batch_data
                rgcn_bg = rgcn_bg.to(device)

                rgcn_node_feats = rgcn_bg.ndata.pop('node').float().to(device)
                rgcn_edge_feats = rgcn_bg.edata.pop('edge').long().to(device)
                smask_feats = rgcn_bg.ndata.pop('smask').unsqueeze(dim=1).float().to(device)
                
                preds, _ = self.model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)   # contains dummy_smiles
                preds = preds[1:]
                preds_prob = preds.sigmoid()
                preds_prob = th.squeeze(preds_prob)
                # preds_label = preds_prob > 0.5

        return preds_prob

        


def load_graph_from_csv_bin_for_splited(
        homog,
        detailed_information,
        group_data,
        smask_path=None,
        classification=True,
        random_shuffle=True,
        seed=2022
):
    data = group_data
    smiles = data.smiles.values
    group = data.group.to_list()
    # load substructure name
    if 'sub_name' in data.columns.tolist():
        sub_name = data['sub_name']
    else:
        sub_name = ['noname' for x in group]

    if random_shuffle:
        random.seed(seed)
        random.shuffle(group)
    labels = detailed_information['labels']

    # load smask
    if smask_path is None:
        smask = [-1 for x in range(len(group))]
    else:
        smask = np.load(smask_path, allow_pickle=True)

    dataset = []
    for index, group_index in enumerate(group):
        molecule = [smiles[index], homog[index], labels[index], smask[index], sub_name[index]]
        dataset.append(molecule)

    return dataset

class Tox_Predictor_Mutag:
    def __init__(self) -> None:

        prop_names = [
            'Mutagenicity-1', 
            'Mutagenicity-2', 
            'Mutagenicity-3', 
            'Mutagenicity-4', 
            'Mutagenicity-5', 
            'Mutagenicity-6', 
            'Mutagenicity-7', 
            'Mutagenicity-8', 
            'Mutagenicity-9', 
            'Mutagenicity-10', 
        ]
        self.configs = [_load_config(n) for n in prop_names]
        self.predictor = []
        for config in self.configs:
            model, config = load_model(config)
            mutag_RGCN_wraper = RGCN_wraper(model, config=config)
            self.predictor.append(mutag_RGCN_wraper)

    def predict(self, smiles_list, device=th.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = th.stack(results)
        column_means = th.mean(results)
        final_prob = column_means.item()

        return final_prob
    
class Tox_Predictor_hERG:
    def __init__(self) -> None:

        prop_names = [
            'hERG-1',
            'hERG-2',
            'hERG-3',
            'hERG-4',
            'hERG-5',
            'hERG-6',
            'hERG-7',
            'hERG-8',
            'hERG-9',
            'hERG-10',
        ]
        self.configs = [_load_config(n) for n in prop_names]
        self.predictor = []
        for config in self.configs:
            model, config = load_model(config)
            mutag_RGCN_wraper = RGCN_wraper(model, config=config)
            self.predictor.append(mutag_RGCN_wraper)

    def predict(self, smiles_list, device=th.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = th.stack(results)
        column_means = th.mean(results)
        final_prob = column_means.item()

        return final_prob
    
class CYP_Predictor_CYP3A4:
    def __init__(self) -> None:

        prop_names = [
            'cyp3a4-1',
            'cyp3a4-2',
            'cyp3a4-3',
            'cyp3a4-4',
            'cyp3a4-5',
            'cyp3a4-6',
            'cyp3a4-7',
            'cyp3a4-8',
            'cyp3a4-9',
            'cyp3a4-10',  
        ]
        self.configs = [_load_config(n) for n in prop_names]
        self.predictor = []
        for config in self.configs:
            model, config = load_model(config)
            mutag_RGCN_wraper = RGCN_wraper(model, config=config)
            self.predictor.append(mutag_RGCN_wraper)

    def predict(self, smiles_list, device=th.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = th.stack(results)
        column_means = th.mean(results)
        final_prob = column_means.item()

        return final_prob
    
class CYP_Predictor_CYP2C19:
    def __init__(self) -> None:

        prop_names = [
            'cyp2c19-1',
            'cyp2c19-2',
            'cyp2c19-3',
            'cyp2c19-4',
            'cyp2c19-5',
            'cyp2c19-6',
            'cyp2c19-7',
            'cyp2c19-8',
            'cyp2c19-9',
            'cyp2c19-10',
        ]
        self.configs = [_load_config(n) for n in prop_names]
        self.predictor = []
        for config in self.configs:
            model, config = load_model(config)
            mutag_RGCN_wraper = RGCN_wraper(model, config=config)
            self.predictor.append(mutag_RGCN_wraper)

    def predict(self, smiles_list, device=th.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = th.stack(results)
        column_means = th.mean(results)
        final_prob = column_means.item()

        return final_prob