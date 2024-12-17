import json
import logging
from pathlib import Path
from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import SyntheticTreeSet

logger = logging.getLogger(__name__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        default='../../../data/molecular_optimization/pre-process/syntrees/filtered_synthetic_trees.json.gz',
        type=str,
        help="Input file for the filtered generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--output-dir",
        default='../../../data/molecular_optimization/pre-process/syntrees/',
        type=str,
        help="Output directory for the splitted synthetic trees (*.json.gz)",
    )

    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    logger.info("Start.")

    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    logger.info(f"Reading data from {args.input_file}")
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    syntrees = syntree_collection.sts

    num_total = len(syntrees)
    logger.info(f"There are {len(syntrees)} synthetic trees.")

    SPLIT_RATIO = [0.6, 0.2, 0.2]

    num_train = int(SPLIT_RATIO[0] * num_total)
    num_valid = int(SPLIT_RATIO[1] * num_total)
    num_test = num_total - num_train - num_valid

    data_train = syntrees[:num_train]
    data_valid = syntrees[num_train : num_train + num_valid]
    data_test = syntrees[num_train + num_valid :]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving training dataset. Number of syntrees: {len(data_train)}")
    SyntheticTreeSet(data_train).save(out_dir / "synthetic-trees-filtered-train.json.gz")

    logger.info(f"Saving validation dataset. Number of syntrees: {len(data_valid)}")
    SyntheticTreeSet(data_valid).save(out_dir / "synthetic-trees-filtered-valid.json.gz")

    logger.info(f"Saving testing dataset. Number of syntrees: {len(data_test)}")
    SyntheticTreeSet(data_test).save(out_dir / "synthetic-trees-filtered-test.json.gz")

    logger.info(f"Completed.")
