from behavior_base import *
from behavioral_pipeline import GoNogoBehaviorMat
import h5py
import numpy as np
import os
import pandas as pd
import re
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')


root_dir = r'X:\HongliWang\Madeline'
raw_beh = 'processed_behavior'
raw_fluo = 'raw_imaging'

# specify saved files
analysis_dir = 'analysis'
analysis_beh = 'behavior'

animals = os.listdir(os.path.join(root_dir,raw_beh))

# initialize the dataframe
columns = ['file','file_path','date', 'subject', 'age', 'saved_dir']
beh_df = pd.DataFrame(columns=columns)

# go through the files to update the dataframe
for animal in animals:
    animal_path = os.path.join(root_dir, raw_beh, animal)
    sessions = glob.glob(os.path.join(animal_path, animal + '*'+'-behaviorLOG.mat'))
    Ind = 0
    for session in sessions:
        separated = os.path.basename(session).split("-")
        data = pd.DataFrame({
            'file': os.path.basename(session),
            'file_path': session,
            'date': separated[1],
            'subject': animal,
            'age': animal[0:3],
            'saved_dir': os.path.join(root_dir, analysis_dir, analysis_beh, animal,separated[1])
        },index=[Ind])
        Ind = Ind + 1
        beh_df = pd.concat([beh_df, data])

nFiles = len(beh_df['file'])
for f in tqdm(range(nFiles)):
    animal = beh_df.iloc[f]['subject']
    session = beh_df.iloc[f]['date']
    input_path = beh_df.iloc[f]['file_path']
    x = GoNogoBehaviorMat(animal, session, input_path)
    x.to_df()
    output_path= beh_df.iloc[f]['saved_dir']
    plot_path = os.path.join(output_path, 'beh_plot')

    # run analysis_beh
    x.d_prime()

    # make plot
    x.beh_session(plot_path)
    x.psycho_curve(plot_path)
    x.lick_rate(plot_path)
    x.ITI_distribution(plot_path)
    x.response_time(plot_path)
    x.running_aligned('onset', plot_path)
    x.running_aligned('outcome',plot_path)
    x.running_aligned('licks',plot_path)

    plt.close('all')
    x.save_analysis(output_path)


# summary plot

# create a dictionary to save summary data

