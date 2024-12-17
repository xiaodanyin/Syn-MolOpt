import json
import logging
from pathlib import Path
from synnet.utils.prep_utils import split_data_into_Xy
from synnet.data_generation.preprocessing import (
    ReactionTemplateFileHandler,
)
logger = logging.getLogger(__file__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default='../../../data/molecular_optimization/featurized/',
        type=str,
        help="Input directory for the featurized synthetic trees (with {train,valid,test}-data).",
    )

    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default="../../../data/molecular_optimization/assets/reaction-templates/rxn_tpl_with_de-tox.txt",  
        help="Input file with reaction templates including the functional templates (de-tox: toxicity-related; de-cyp: metabolism-related).",
    )

    return parser.parse_args()

if __name__ == "__main__":
    logger.info("Start.")

    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    logger.info("Start splitting data.")

    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    num_rxn = len(rxn_templates)  
    out_dim = 256 
    input_dir = Path(args.input_dir)
    output_dir = input_dir / "Xy"
    for dataset_type in "train valid test".split():
        logger.info(f"Split {dataset_type}-data...")
        split_data_into_Xy(
            dataset_type=dataset_type,
            steps_file=input_dir / f"{dataset_type}_steps.npz",
            states_file=input_dir / f"{dataset_type}_states.npz",
            output_dir=input_dir / "Xy",
            num_rxn=num_rxn,
            out_dim=out_dim,
        )

    logger.info(f"Completed.")
