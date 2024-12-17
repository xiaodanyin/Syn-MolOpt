import logging
from rdkit import RDLogger
from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from synnet.utils.data_utils import ReactionSet

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
import json


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--building-blocks-file",
        default='../../../data/molecular_optimization/assets/building-blocks/preprocessed_building_blocks.csv',
        type=str,
        help="File with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        default='../../../data/molecular_optimization/assets/reaction-templates/rxn_tpl_with_de-tox.txt',
        type=str,
        help="Input file with reaction templates including the functional templates (de-tox: toxicity-related; de-cyp: metabolism-related).",
    )
    parser.add_argument(
        "--output-bblock-file",
        default='../../../data/molecular_optimization/pre-process/building-blocks-rxns/filtered_building_block.csv.gz',
        type=str,
        help="Output file for the filtered building-blocks.",
    )
    parser.add_argument(
        "--output-rxns-collection-file",
        default='../../../data/molecular_optimization/pre-process/building-blocks-rxns/filtered_rxn_tpl_with_de-tox.json.gz',
        type=str,
        help="Output file for the collection of reactions matched with building-blocks.",
    )
    
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    bbf = BuildingBlockFilter(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        verbose=args.verbose,
        processes=args.ncpu,
    )
    bbf.filter()

    bblocks_filtered = bbf.building_blocks_filtered
    BuildingBlockFileHandler().save(args.output_bblock_file, bblocks_filtered)

    rxn_collection = ReactionSet(bbf.rxns)
    rxn_collection.save(args.output_rxns_collection_file)

    logger.info(f"Total number of building blocks {len(bblocks):d}")
    logger.info(f"Matched number of building blocks {len(bblocks_filtered):d}")
    logger.info(
        f"{len(bblocks_filtered)/len(bblocks):.2%} of building blocks applicable for the reaction templates."
    )

    logger.info("Completed.")
