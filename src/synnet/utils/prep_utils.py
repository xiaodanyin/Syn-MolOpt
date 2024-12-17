import logging
from pathlib import Path
from typing import Iterator, Union
import numpy as np
from rdkit import Chem
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

def rdkit2d_embedding(smi):
    """
    Computes an embedding using RDKit 2D descriptors.

    Args:
        smi (str): SMILES string.

    Returns:
        np.ndarray: A molecular embedding corresponding to the input molecule.
    """
    from tdc.chem_utils import MolConvert

    if smi is None:
        return np.zeros(200).reshape((-1,))
    else:
        # define the RDKit 2D descriptor
        rdkit2d = MolConvert(src="SMILES", dst="RDKit2D")
        return rdkit2d(smi).reshape(
            -1,
        )

import functools

@functools.lru_cache(maxsize=1)
def _fetch_gin_pretrained_model(model_name: str):
    from dgllife.model import load_pretrained

    """Get a GIN pretrained model to use for creating molecular embeddings"""
    device = "cpu"
    model = load_pretrained(model_name).to(device)
    model.eval()
    return model

def split_data_into_Xy(
    dataset_type: str,
    steps_file: str,
    states_file: str,
    output_dir: Path,
    num_rxn: int,
    out_dim: int,
) -> None:
    """Split the featurized data into X,y-chunks for the {act,rt1,rxn,rt2}-networks.

    Args:
        num_rxn (int): Number of reactions in the dataset.
        out_dim (int): Size of the output feature vectors (used in kNN-search for rt1,rt2)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    states = sparse.load_npz(states_file)  # (n,3*4096)
    steps = sparse.load_npz(steps_file)  # (n,1+256+1+256+4096)
    states = states.tocsc()
    steps = steps.tocsc()

    # Extract data for each network...      
  
    X = states                
    y = steps[:, 0]
    sparse.save_npz(output_dir / f"X_act_{dataset_type}.npz", X)
    sparse.save_npz(output_dir / f"y_act_{dataset_type}.npz", y)
    logger.info(f'  saved data for "Action" to {output_dir}')

    actions = steps[:, 0].A  
    isActionEnd = (actions == 3).squeeze()  
    states = states[~isActionEnd]
    steps = steps[~isActionEnd]
    X = sparse.hstack([states, steps[:, (2 * out_dim + 2) :]])  
    y = steps[:, out_dim + 1] 
    sparse.save_npz(output_dir / f"X_rxn_{dataset_type}.npz", X)
    sparse.save_npz(output_dir / f"y_rxn_{dataset_type}.npz", y)
    logger.info(f'  saved data for "Reaction" to {output_dir}')

    actions = steps[:, 0].A  
    isActionMerge = (actions == 2).squeeze()  
    steps = steps[~isActionMerge]
    states = states[~isActionMerge]
    z_mol1 = steps[:, (2 * out_dim + 2) :]
    rxn_ids = steps[:, (1 + out_dim)]
    z_rxn_id = OneHotEncoder().fit(np.arange(num_rxn)[:, None]).transform(rxn_ids.A)
    X = sparse.hstack((states, z_mol1, z_rxn_id))  
    y = steps[:, (2 + out_dim) : (2 * out_dim + 2)]
    sparse.save_npz(output_dir / f"X_rt2_{dataset_type}.npz", X)
    sparse.save_npz(output_dir / f"y_rt2_{dataset_type}.npz", y)
    logger.info(f'  saved data for "Reactant 2" to {output_dir}')

    actions = steps[:, 0].A  
    isActionExpand = (actions == 1).squeeze() 
    steps = steps[~isActionExpand]
    states = states[~isActionExpand]
    zprime_mol1 = steps[:, 1 : (out_dim + 1)]

    X = states
    y = zprime_mol1
    sparse.save_npz(output_dir / f"X_rt1_{dataset_type}.npz", X)
    sparse.save_npz(output_dir / f"y_rt1_{dataset_type}.npz", y)
    logger.info(f'  saved data for "Reactant 1" to {output_dir}')

    return None

class Sdf2SmilesExtractor:
    """Helper class for data generation."""

    def __init__(self) -> None:
        self.smiles: Iterator[str]

    def from_sdf(self, file: Union[str, Path]):
        """Extract chemicals as SMILES from `*.sdf` file.

        See also:
            https://www.rdkit.org/docs/GettingStartedInPython.html#reading-sets-of-molecules
        """
        file = str(Path(file).resolve())
        suppl = Chem.SDMolSupplier(file)
        self.smiles = (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in suppl if mol is not None)
        logger.info(f"Read data from {file}")

        return self

    def _to_csv_gz(self, file: Path) -> None:
        import gzip

        with gzip.open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def _to_txt(self, file: Path) -> None:
        with open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def to_file(self, file: Union[str, Path]) -> None:

        if Path(file).suffixes == [".csv", ".gz"]:
            self._to_csv_gz(file)
        else:
            self._to_txt(file)
        logger.info(f"Saved data to {file}")
