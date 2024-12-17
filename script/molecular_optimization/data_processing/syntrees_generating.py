import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict
from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from synnet.data_generation.syntrees import SynTreeGenerator, wraps_syntreegenerator_generate
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet

logger = logging.getLogger(__name__)
from typing import Tuple, Union

RDLogger.DisableLog("rdApp.*")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--building-blocks-file",
        type=str,
        default="../../../data/molecular_optimization/pre-process/building-blocks-rxns/filtered_building_block.csv.gz",  
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default="../../../data/molecular_optimization/assets/reaction-templates/rxn_tpl_with_de-tox.txt",  
        help="Input file with reaction templates including the functional templates (de-tox: toxicity-related; de-cyp: metabolism-related).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="../../../data/molecular_optimization/pre-process/syntrees/synthetic_trees.json.gz",
        help="Output file for the generated synthetic trees (*.json.gz)",
    )

    parser.add_argument(
        "--number-syntrees", type=int, default=600000, help="Number of SynTrees to generate.",
    )
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def generate_mp() -> Tuple[Dict[int, str], List[Union[SyntheticTree, None]]]:
    from functools import partial
    import numpy as np
    from pathos import multiprocessing as mp

    def wrapper(stgen, _):
        stgen.rng = np.random.default_rng() 
        return wraps_syntreegenerator_generate(stgen)

    func = partial(wrapper, stgen)

    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, range(args.number_syntrees))

    outcomes = {
        i: e.__class__.__name__ if e is not None else "success" for i, (_, e) in enumerate(results)
    }
    syntrees = [st for (st, e) in results if e is None]
    return outcomes, syntrees


def generate() -> Tuple[Dict[int, str], List[Union[SyntheticTree, None]]]:
    outcomes: Dict[int, str] = dict()
    syntrees: List[Union[SyntheticTree, None]] = []
    myrange = tqdm(range(args.number_syntrees)) if args.verbose else range(args.number_syntrees)
    for i in myrange:
        st, e = wraps_syntreegenerator_generate(stgen)
        outcomes[i] = e.__class__.__name__ if e is not None else "success"
        syntrees.append(st)

    return outcomes, syntrees

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    logger.info("Start.")

    args = get_args()
    seed = 42
    seed_torch(seed)

    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    public_templates = rxn_templates[:91]
    functional_templates = rxn_templates[91: len(rxn_templates)]
    logger.info("Loaded building block & rxn-template assets.")

    logger.info("Start initializing SynTreeGenerator...")
    stgen = SynTreeGenerator(
        building_blocks=bblocks, 
        rxn_templates=public_templates, 
        functional_template = functional_templates, 
        rng = np.random.default_rng(seed=seed), 
        verbose=args.verbose
    )       
    logger.info("Successfully initialized SynTreeGenerator.")

    logger.info(f"Start generation of {args.number_syntrees} SynTrees...")
    if args.ncpu > 1:
        outcomes, syntrees = generate_mp()
    else:
        outcomes, syntrees = generate()
    result_summary = Counter(outcomes.values())
    logger.info(f"SynTree generation completed. Results: {result_summary}")

    summary_file = Path(args.output_file).parent / "results_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(result_summary, indent=2))

    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection.save(args.output_file)

    logger.info(f"Completed.")
