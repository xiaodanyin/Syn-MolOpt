import json
import logging
from functools import partial
from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.fingerprints import mol_fp
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__file__)

FUNCTIONS = {
    "fp_4096": partial(mol_fp, _radius=2, _nBits=4096),
    "fp_2048": partial(mol_fp, _radius=2, _nBits=2048),
    "fp_1024": partial(mol_fp, _radius=2, _nBits=1024),
    "fp_512": partial(mol_fp, _radius=2, _nBits=512),
    "fp_256": partial(mol_fp, _radius=2, _nBits=256),
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--building-blocks-file",
        default='../../../data/molecular_optimization/pre-process/building-blocks-rxns/filtered_building_block.csv.gz',
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--output-file",
        default='../../../data/molecular_optimization/pre-process/embeddings/filtered_building_block_embeddings.npy',
        type=str,
        help="Output file for the computed embeddings file. (*.npy)",
    )
    parser.add_argument(
        "--featurization-fct",
        default='fp_256',
        type=str,
        choices=FUNCTIONS.keys(),
        help="Featurization function applied to each molecule.",
    )
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    logger.info(f"Successfully read {args.building_blocks_file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")        

    func = FUNCTIONS[args.featurization_fct]
    molembedder = MolEmbedder(processes=args.ncpu).compute_embeddings(func, bblocks)
    molembedder.save_precomputed(args.output_file)

    logger.info("Completed.")
