from generate_features_epieco import *
import os

from parsing import EpiEcoParser

base_folder = "./../data/folds"
data_parser = EpiEcoParser(**cfg.epieco_parser_args)

# Run only if is desired to generate new folds subsets -- params -> [k=5]
# data_parser,generate_k_folds()
data_parser.generate_k_folds_features()



