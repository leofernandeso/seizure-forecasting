import numpy as np
import pandas as pd
import os

# main feature extractor 
from feature_extractor import *

# epilepsy ecosystem parser
from parsing import EpiEcoParser

# visualization
import visualization

def main():
    base_folder = 'D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\Train'
    dropout_filepath = base_folder + "\\drop_out_segments.txt"

    save_plots_folder = "D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\dropouts_imgs"
    if not os.path.exists(save_plots_folder):
        os.mkdir(save_plots_folder)
    else:
        print("DROPOUT PLOTS FOLDER ALREADY EXISTS!!")

    data_parser = EpiEcoParser(**cfg.parser_args)

    with open(dropout_filepath, 'r') as file:
        line = file.readline()
        while line:
            path = os.path.join(base_folder, line.strip())
            channels = data_parser.load_segment_from_path(path)

            file_id = os.path.split(line)[-1].split('.')[0]
            print(file_id)
            visualization.plot_eeg(channels, data_parser.fs, savepath=os.path.join(save_plots_folder, file_id))
            line = file.readline()

    
    

if __name__ == '__main__':
    main()