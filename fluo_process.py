import numpy as np
from os.path import join as oj
from utils_signal import *
import pandas as pd
from tqdm import tqdm
from pyPlotHW import StartPlots, StartSubplots
import matplotlib.pyplot as plt

# class to preprocess calcium 2-photon data

class Suite2pSeries:

    def __init__(self, suite2p):
        suite2p = oj(suite2p, 'plane0')
        Fraw = np.load(oj(suite2p, 'F.npy'))
        ops = np.load(oj(suite2p, 'ops.npy'), allow_pickle=True)
        neuropil = np.load(oj(suite2p, 'Fneu.npy'))
        cells = np.load(oj(suite2p, 'iscell.npy'))
        stat = np.load(oj(suite2p, 'stat.npy'), allow_pickle=True)
        self.Fraw = Fraw
        self.ops = ops
        self.neuropil = neuropil
        self.cells = cells
        self.stat = stat

        F = Fraw - neuropil * 0.7  # subtract neuropil
        # find number of cells
        numcells = np.sum(cells[:, 0] == 1.0)
        # create a new array (Fcells) with F data for each cell
        Fcells = F[cells[:, 0] == 1.0]

        # filter the raw fluorscent data, finding the baseline?
        F0_AQ = np.zeros(Fcells.shape)
        for cell in range(Fcells.shape[0]):
            F0_AQ[cell] = robust_filter(Fcells[cell], method=12, window=200, optimize_window=2, buffer=False)[:, 0]

        dFF = np.zeros(Fcells.shape)
        for cell in tqdm(range(0, Fcells.shape[0])):
            for frame in range(0, Fcells.shape[1]):
                dFF[cell, frame] = (Fcells[cell, frame] - F0_AQ[cell, frame]) / F0_AQ[cell, frame]

        self.neural_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(numcells)])
        self.neural_df['time'] = np.arange(self.neural_df.shape[0])

    def realign_time(self, reference=None):  # need the behavior mat as reference
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)

        if self.neural_df is not None:
            # aligned to time 0
            self.neural_df['time'] = transform_func(self.neural_df['time'])-reference.time_0


    # def calculate_dff(self):
    #     rois = list(self.neural_df.columns[1:])
    #     melted = pd.melt(self.neural_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
    #     return melted
    #
    def calculate_dff(self, method='robust', melt=True): # provides different options for how to calculate dF/F
        # results are wrong
        time_axis = self.neural_df['time']
        if method == 'old':
            Fcells = self.neural_df.values.T
            F0 = []
            for cell in range(0, Fcells.shape[0]):
                include_frames = []
                std = np.std(Fcells[cell])
                avg = np.mean(Fcells[cell])
                for frame in range(0, Fcells.shape[1]):
                    if Fcells[cell, frame] < std + avg:
                        include_frames.append(Fcells[cell, frame])
                F0.append(np.mean(include_frames))
            dFF = np.zeros(Fcells.shape)
            for cell in range(0, Fcells.shape[0]):
                for frame in range(0, Fcells.shape[1]):
                    dFF[cell, frame] = (Fcells[cell, frame] - F0[cell]) / F0[cell]
        elif method == 'robust':
            Fcells = self.neural_df.values.T
            dFF = np.zeros(Fcells.shape) # d
            for cell in tqdm(range(Fcells.shape[0])):
                f0_cell = robust_filter(Fcells[cell], method=12, window=200, optimize_window=2, buffer=False)[:, 0]
                dFF[cell] = (Fcells[cell] - f0_cell) / f0_cell
        dff_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(Fcells.shape[0])])
        dff_df['time'] = time_axis
        if melt:
            rois = [c for c in dff_df.columns if c != 'time']
            melted = pd.melt(dff_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
            return melted
        else:
            return dff_df

    def plot_cell_location_dFF(self, neuron_range):
        import random

        cellstat = []
        for cell in range(0, self.Fraw.shape[0]):
            if self.cells[cell, 0] > 0:
                cellstat.append(self.stat[cell])

        fluoCellPlot = StartPlots()
        im = np.zeros((256, 256))

        for cell in neuron_range:

            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
            im[ys, xs] = random.random()


        fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        plt.show()

        return fluoCellPlot

    def plot_cell_dFF(self, neuron_range):

        fluoTracePlot = StartPlots()
        for cell in neuron_range:
            fluoTracePlot.ax.plot(self.neural_df.iloc[15000:20000, cell] + cell, label="Neuron " + str(cell))
        plt.show()

        return fluoTracePlot






def robust_filter(ys, method=12, window=200, optimize_window=2, buffer=False):
    """
    First 2 * windows re-estimate with mode filter
    To avoid edge effects as beginning, it uses mode filter; better solution: specify initial conditions
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=buffer)
    opt_w = int(np.rint(optimize_window * window))
    # prepend
    init_win_ys = ys[:opt_w]
    prepend_ys = init_win_ys[opt_w-1:0:-1]
    ys_pp = np.concatenate([prepend_ys, ys])
    f0 = np.array([(mf(ys_pp, i), mDC.get_dev()) for i in range(len(ys_pp))])[opt_w-1:]
    return f0

if __name__ == "__main__":
    #input_folder = r"Z:\Madeline\processed_data\JUV015\220409\suite2p"
    input_folder = r"C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\suite2p_output"
    gn_series = Suite2pSeries(input_folder)
    animal, session = 'JUV011', '211215'
    # dff_df = gn_series.calculate_dff(melt=False)

    # get behavior
    from behavioral_pipeline import BehaviorMat, GoNogoBehaviorMat
    #beh_folder = "C:\\Users\\hongl\\Documents\\GitHub\\madeline_go_nogo\\data"
    beh_folder = "C:\\Users\\xiachong\\Documents\\GitHub\\madeline_go_nogo\\data"
    beh_file = "JUV015_220409_behaviorLOG.mat"
    trialbytrial = GoNogoBehaviorMat(animal, session, oj(beh_folder, beh_file))

    # align behavior with fluorescent data
    gn_series.realign_time(trialbytrial)

    # save file
    gn_series.neural_df.to_csv(r"C:\Users\xiachong\Documents\GitHub\JUV015_220409_dff_df_file.csv")
    gn_series.plot_cell_location_dFF(np.arange(gn_series.neural_df.shape[1]-1))
    x= 1