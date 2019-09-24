from generate_features import *
import os

from parsing import EpiEcoParser

base_folder = "D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\folds"
data_parser = EpiEcoParser(**cfg.parser_args)

for dir in os.listdir(base_folder):
    fold_folder = os.path.join(base_folder, dir)
    for fold_file in os.listdir(fold_folder):
        abs_path = os.path.join(fold_folder, fold_file)
        df = pd.read_csv(abs_path)
        dest = abs_path.split('.')[0]+'_features.csv'
        generate_features(df, data_parser, dest, cfg.train_drop_out_filepath, cfg.join_windows)      
        