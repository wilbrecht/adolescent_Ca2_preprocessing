# .py file to run behavior analysis

from behavior_base import *
from behavioral_pipeline import GoNogoBehaviorMat
import h5py
import numpy as np
import os


# specify example data session
# to do: go through the whole data set and generate analysis result for each session
# maybe wrap this in a single class?

animal = 'JUV015'
session = '220409'

input_folder = "C:\\Users\\hongl\\Documents\\GitHub\\madeline_go_nogo\\data"
#input_folder = "C:\\Users\\xiachong\\Documents\\GitHub\\madeline_go_nogo\\data"
input_file = "JUV015_220409_behaviorLOG.mat"
hfile = h5py.File(os.path.join(input_folder, input_file), 'r')
hfile['out'].keys() # prints trial variables

# format trial events
eventlist = EventNode(None, None, None, None)
code_map = GoNogoBehaviorMat.code_map
trial_events = np.array(hfile['out/GoNG_EventTimes'])
running_speed = np.array(hfile['out/run_speed'])
frame_time = np.array(hfile['out/frame_time'])

trialbytrial = GoNogoBehaviorMat(animal, session, os.path.join(input_folder, input_file))
result_df = trialbytrial.to_df()

# save data
result_df.to_csv(os.path.join(input_folder, f"{animal}_{session}_behavior_output.csv"))

# generate analysis plots

saveBehFigPath = os.path.join(input_folder, 'beh_plot')

# session summary
behPlot = trialbytrial.beh_session()
behPlot.save_plot('Behave summary.svg', 'svg', saveBehFigPath)
behPlot.save_plot('Behave summary.tif', 'tif', saveBehFigPath)

# psychometric curve
psyPlot = trialbytrial.psycho_curve()
psyPlot.save_plot('Psychometric curve.svg', 'svg', saveBehFigPath)
psyPlot.save_plot('Psychometric curve.tif', 'tif', saveBehFigPath)

#
rtPlot = trialbytrial.response_time()
rtPlot.save_plot('Response time.svg', 'svg', saveBehFigPath)
rtPlot.save_plot('Response time.tif', 'tif', saveBehFigPath)

# plot lick rate for Hit/false alarm trials
lickRatePlot = trialbytrial.lick_rate()
lickRatePlot.save_plot('Lick rate.svg', 'svg', saveBehFigPath)
lickRatePlot.save_plot('Lick rate.tif', 'tif', saveBehFigPath)

# ITI plot
ITIPlot = trialbytrial.ITI_distribution()
ITIPlot.save_plot('ITI distribution.svg', 'svg', saveBehFigPath)
ITIPlot.save_plot('ITI distribution.tif', 'tif', saveBehFigPath)

# running speed plot
runPlot1 = trialbytrial.running_aligned('onset')
runPlot1.save_plot('Run_aligned_onset.svg', 'svg', saveBehFigPath)
runPlot1.save_plot('Run_aligned_onset.tif', 'tif', saveBehFigPath)

runPlot2 = trialbytrial.running_aligned('outcome')
runPlot2.save_plot('Run_aligned_outcome.svg', 'svg', saveBehFigPath)
runPlot2.save_plot('Run_aligned_outcome.tif', 'tif', saveBehFigPath)

runPlot3 = trialbytrial.running_aligned('licks')
runPlot3.save_plot('Run_aligned_licks.svg', 'svg', saveBehFigPath)
runPlot3.save_plot('Run_aligned_licks.tif', 'tif', saveBehFigPath)

