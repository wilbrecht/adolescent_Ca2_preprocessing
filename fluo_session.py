# .py file to read in the behavior result, and combine it with calcium imaging result

from behavior_base import *
from behavioral_pipeline import GoNogoBehaviorMat
import h5py
import numpy as np
import os
import pandas as pd

animal = 'JUV015'
session = '220409'

input_folder = "C:\\Users\\hongl\\Documents\\GitHub\\madeline_go_nogo\\data"
#input_folder = "C:\\Users\\xiachong\\Documents\\GitHub\\madeline_go_nogo\\data"
input_file = "JUV015_220409_behaviorLOG.mat"
output_file = f"{animal}_{session}_behavior_output.csv"

beh_df = pd.read_csv (os.path.join(input_folder, output_file))

# preprocess imaging data in fluo_process

# start with df/f here

