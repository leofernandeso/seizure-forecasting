import os
import shutil


dropout_path = "D:\Faculdade\TCC\dados\epilepsy_ecosystem\Train\drop_out_segments.txt"
destination_folder = "D:\Faculdade\TCC\dados\epilepsy_ecosystem\dropouts_files"

with open(dropout_path, 'r') as dropout_file:
    line = dropout_file.readline().rstrip()
    while line:
        shutil.move(line, destination_folder)
        line = dropout_file.readline().rstrip()
