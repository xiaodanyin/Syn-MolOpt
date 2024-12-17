# Syn-MolOpt
Syn-MolOpt: A Synthesis Planning-Driven Molecular Optimization Method Using Data-Derived Functional Reaction Templates

## Contents
- [Installation Guide](#installation-guide)
- [Data Download](#data-download)
- [Running Syn-MolOpt](#running-Syn-MolOpt)

## Installation Guide
Create a virtual environment to run the code of Syn-MolOpt.<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/xiaodanyin/Syn-MolOpt.git.
cd Syn-MolOpt
conda env create -f envs.yaml
conda activate syn_molopt
pip install -e .
```

## Data Download

The **trained_models** and **data** can be downloaded from the following link: [https://drive.google.com/file/d/1_d-Lw9qksr4Y1ZXbJdV2jEwLwUWK0lBV/view?usp=sharing](https://drive.google.com/file/d/1_d-Lw9qksr4Y1ZXbJdV2jEwLwUWK0lBV/view?usp=sharing).

The **building blocks** are not freely available. To obtain the data, please visit [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog), and we utilized the "Building Blocks, US Stock" data for our work.

## Running Syn-MolOpt

Data processing, and model implementation methods for functional substructure analysis, reaction template extraction and molecular optimization are provided at `scripts/`.

To perform molecular optimization, enter the following command:<br>

```
cd script/molecular_optimization
python main.py  --building-blocks-file path/to/building_blocks_file \
                --rxns-collection-file path/to/rxns_collection_file \
                --embeddings-knn-file path/to/embeddings_knn_file \
                --ckpt-dir path/to/checkpoints/ \
                --input-file path/to/input.csv \
                --output_file path/to/output_file \
                --output_trees_file path/to/output_trees_file \
```

