import pyvista
import math
import numpy as np
from my_write_ply_file_mirror import my_write_ply_file
from pathlib import Path
import cv2
import glob
from tqdm import tqdm
path_in = '/media/lau/My Passport/data/Derma2021/dense_dema_aligned/'
path_out = '/media/lau/My Passport/data/Derma2021/dense_dema_aligned_mirror/'


ply_file_list = glob.glob(path_in + '*.ply')


for i_total, data_file in tqdm(enumerate(ply_file_list)):
    print(data_file)
    mesh_original = pyvista.read(data_file)
    ply_file_out = data_file.replace(path_in,path_out)
    my_write_ply_file(mesh_original, ply_file_out)
