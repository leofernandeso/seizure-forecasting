import numpy as np
import pandas as pd
import pickle
import time

# main feature extractor 
from feature_extractor import generate_features

# epilepsy ecosystem parser
from parsing import EpiEcoParser

# parsing config
import feature_extractor_config as cfg

def main():

    train_path = "C:\\Users\\Leonardo\\Documents\\Faculdade\\TCC\\processed_data\\subsets\\train.p"
    val_path = "C:\\Users\\Leonardo\\Documents\\Faculdade\\TCC\\processed_data\\subsets\\validation.p"

    data_parser = EpiEcoParser(**cfg.epieco_parser_args)

    # Run this line only to generate new subsets!
    df_train, df_val = data_parser.process_dataset()


    with open(train_path, 'rb') as df_file:
            df_train = pickle.load(df_file)
    with open(val_path, 'rb') as df_file:
            df_val = pickle.load(df_file)

    
    #generate_features(df_train, data_parser, cfg.csv_train_output_path, cfg.train_drop_out_filepath, cfg.join_windows)      
    #generate_features(df_val, data_parser, cfg.csv_val_output_path, cfg.train_drop_out_filepath, cfg.join_windows)      


    

if __name__ == '__main__':
    main()