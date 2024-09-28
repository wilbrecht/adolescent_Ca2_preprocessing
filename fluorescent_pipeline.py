# pipeline for fluorescent analysis
# %matplotlib inline
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from pyPlotHW import *
from utility_HW import bootstrap, count_consecutive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from scipy.stats import binomtest
from behavioral_pipeline import BehaviorMat, GoNogoBehaviorMat
import os
from utils_signal import *
import pickle
from packages.decodanda_master.decodanda import Decodanda
import seaborn as sns

from scipy.stats import wilcoxon
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import random

import matlab.engine

warnings.filterwarnings("ignore")
# read df/f and behavior data, create a class with behavior and df/f data
class Suite2pSeries:

    def __init__(self, suite2p):
        suite2p = os.path.join(suite2p, 'plane0')
        Fraw = np.load(os.path.join(suite2p, 'F.npy'))
        ops = np.load(os.path.join(suite2p, 'ops.npy'), allow_pickle=True)
        neuropil = np.load(os.path.join(suite2p, 'Fneu.npy'))
        cells = np.load(os.path.join(suite2p, 'iscell.npy'))
        stat = np.load(os.path.join(suite2p, 'stat.npy'), allow_pickle=True)
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
        print("Calculating dFFs........")
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

class fluoAnalysis:

    def __init__(self, beh_file, fluo_file):
        # check file type
        if beh_file.partition('.')[2] == 'csv':
            self.beh = pd.read_csv(beh_file)
        elif beh_file.partition('.')[2] == 'pickle':
            with open(beh_file, 'rb') as pf:
                # Load the data from the pickle file
                beh_data = pickle.load(pf)
                pf.close()
            self.beh = beh_data['behDF']
        self.fluo = pd.read_csv(fluo_file)

    def align_fluo_beh(self):
        # align the fluorescent data to behavior based on trials
        # interpolate fluorescent data and running speed
        nTrials = self.beh.shape[0]
        nCells = self.fluo.shape[1] - 2 # exclude the first column (index) and last column (time)
        startTime = -3
        endTime = 8
        step = 0.01
        interpT = np.arange(startTime, endTime, step)
        self.dFF_aligned = np.zeros((len(interpT), nTrials, nCells))
        self.run_aligned = np.zeros((len(interpT), nTrials))
        self.interpT = interpT

        for tt in range(nTrials):

            # align speed
            speed = self.beh['running_speed'][tt]
            #speed = np.array(eval(speed))
            speed = np.array(speed)
            speed_time = self.beh['running_time'][tt]
            #speed_time = np.array(eval(speed_time))
            speed_time = np.array(speed_time)
            if speed.size != 0:
                t = speed_time - self.beh['onset'][tt]
                y = speed
                self.run_aligned[:, tt] = np.interp(interpT, t, y)

            # align dFF
            tempStart = self.beh['onset'][tt] + startTime
            tempEnd = self.beh['onset'][tt] + endTime
            t_dFF = self.fluo['time'].values
            for cc in range(nCells):
                dFF = self.fluo['neuron'+str(cc)].values
                # get the trial timing
                tempT = t_dFF[np.logical_and(t_dFF>tempStart, t_dFF<=tempEnd)]-self.beh['onset'][tt]
                tempdFF = dFF[np.logical_and(t_dFF>tempStart, t_dFF<=tempEnd)]
                # interpolate
                self.dFF_aligned[:,tt,cc] = np.interp(interpT, tempT, tempdFF)

    def plot_dFF(self, savefigpath):
        # PSTH plot for different trial types
        # cue
        nCells = self.fluo.shape[1]-2
        goCue = [1, 2, 3, 4]
        noGoCue = [5, 6, 7, 8]
        probeGoCue = [9, 10, 11, 12]
        probeNoGoCue = [13, 14, 15, 16]

        goTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in goCue]
        noGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in noGoCue]
        probeGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in probeGoCue]
        probeNoGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in probeNoGoCue]

        for cc in tqdm(range(nCells)):

            # get dFF in trials and bootstrap

            dFFPlot = StartSubplots(2,2, ifSharey=True)

            # subplot 1: dFF traces of different cues

    ### plotting PSTH for go/nogo/probe cues--------------------------------------------
            tempGodFF = self.dFF_aligned[:, goTrials, cc]
            bootGo = bootstrap(tempGodFF, 1, 1000)

            tempNoGodFF = self.dFF_aligned[:, noGoTrials, cc]
            bootNoGo = bootstrap(tempNoGodFF, 1, 1000)

            tempProbeGodFF = self.dFF_aligned[:, probeGoTrials, cc]
            bootProbeGo = bootstrap(tempProbeGodFF, 1, 1000)

            tempProbeNoGodFF = self.dFF_aligned[:, probeNoGoTrials, cc]
            bootProbeNoGo = bootstrap(tempProbeNoGodFF, 1, 1000)

            dFFPlot.fig.suptitle('Cell ' + str(cc+1))

            dFFPlot.ax[0,0].plot(self.interpT, bootGo['bootAve'], color=(1,0,0))
            dFFPlot.ax[0,0].fill_between(self.interpT, bootGo['bootLow'], bootGo['bootHigh'], color = (1,0,0),label='_nolegend_', alpha=0.2)

            dFFPlot.ax[0, 0].plot(self.interpT, bootNoGo['bootAve'], color=(0, 1, 0))
            dFFPlot.ax[0, 0].fill_between(self.interpT, bootNoGo['bootLow'], bootNoGo['bootHigh'], color=(0, 1, 0),label='_nolegend_',
                                          alpha=0.2)
            dFFPlot.ax[0,0].legend(['Go', 'No go'])
            dFFPlot.ax[0,0].set_title('Cue')
            dFFPlot.ax[0,0].set_ylabel('dFF')

            dFFPlot.ax[0, 1].plot(self.interpT, bootProbeGo['bootAve'], color=(1, 0, 0))
            dFFPlot.ax[0, 1].fill_between(self.interpT, bootProbeGo['bootLow'], bootProbeGo['bootHigh'], color=(1, 0, 0),label='_nolegend_',
                                          alpha=0.2)

            dFFPlot.ax[0, 1].plot(self.interpT, bootProbeNoGo['bootAve'], color=(0, 1, 0))
            dFFPlot.ax[0, 1].fill_between(self.interpT, bootProbeNoGo['bootLow'], bootProbeNoGo['bootHigh'], color=(0, 1, 0),label='_nolegend_',
                                          alpha=0.2)

            dFFPlot.ax[0, 1].legend(['Probe go', 'Probe no go'])
            dFFPlot.ax[0, 1].set_title('Cue')
            dFFPlot.ax[0, 1].set_ylabel('dFF')

    ### this part is used to plot PSTH for every individual cues
    ### ------------------------------------------------------------------------------
            # cues = np.unique(self.beh['sound_num'])
            #
            # for cue in cues:
            #
            #     tempdFF = self.dFF_aligned[:,self.beh['sound_num']==cue,cc]
            #     bootTemp = bootstrap(tempdFF, 1, 1000)
            #     dFFPlot.fig.suptitle('Cell ' + str(cc+1))
            #
            #     # set color
            #     if cue <= 4:
            #         c = (1, 50*cue/255, 50*cue/255)
            #         subInd = 0
            #     elif cue > 4 and cue<=8:
            #         c = (50*(cue-4)/255, 1, 50*(cue-4)/255)
            #         subInd = 0
            #     else:
            #         c = (25*(cue-8)/255, 25*(cue-8)/255, 1)
            #         subInd = 1
            #
            #     dFFPlot.ax[0,subInd].plot(self.interpT, bootTemp['bootAve'], color=c)
            #     dFFPlot.ax[0,subInd].fill_between(self.interpT, bootTemp['bootLow'], bootTemp['bootHigh'], color = c, alpha=0.2)
            #     dFFPlot.ax[0,subInd].set_title('Cue')
            #     dFFPlot.ax[0,subInd].set_ylabel('dFF')
    ###------------------------------------------------------------------------------------------------------

            Hit_dFF = self.dFF_aligned[:,self.beh['choice']==2,cc]
            FA_dFF = self.dFF_aligned[:, self.beh['choice'] == -1, cc]
            Miss_dFF = self.dFF_aligned[:, self.beh['choice'] == -2, cc]
            CorRej_dFF = self.dFF_aligned[:, self.beh['choice'] == 0, cc]
            ProbeLick_dFF = self.dFF_aligned[:, self.beh['choice'] == -3, cc]
            ProbeNoLick_dFF = self.dFF_aligned[:, self.beh['choice'] == -4, cc]

            Hit_boot = bootstrap(Hit_dFF, 1, 1000)
            FA_boot = bootstrap(FA_dFF, 1, 1000)
            Miss_boot = bootstrap(Miss_dFF, 1, 1000)
            CorRej_boot = bootstrap(CorRej_dFF, 1, 1000)
            ProbeLick_boot = bootstrap(ProbeLick_dFF, 1, 1000)
            ProbeNoLick_boot = bootstrap(ProbeNoLick_dFF, 1, 1000)

            # get cmap
            cmap = matplotlib.colormaps['jet']

            dFFPlot.ax[1, 0].plot(self.interpT, Hit_boot['bootAve'], color = cmap(0.1))
            dFFPlot.ax[1, 0].fill_between(self.interpT, Hit_boot['bootLow'], Hit_boot['bootHigh'],
                                               alpha=0.2, label='_nolegend_', color = cmap(0.1))
            dFFPlot.ax[1, 0].plot(self.interpT, FA_boot['bootAve'], color = cmap(0.3))
            dFFPlot.ax[1, 0].fill_between(self.interpT, FA_boot['bootLow'], FA_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_',color = cmap(0.3))
            dFFPlot.ax[1, 0].plot(self.interpT, Miss_boot['bootAve'], color = cmap(0.5))
            dFFPlot.ax[1, 0].fill_between(self.interpT, Miss_boot['bootLow'], Miss_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.5))
            dFFPlot.ax[1, 0].plot(self.interpT, CorRej_boot['bootAve'], color = cmap(0.7))
            dFFPlot.ax[1, 0].fill_between(self.interpT, CorRej_boot['bootLow'], CorRej_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.7))
            dFFPlot.ax[1, 0].legend(['Hit', 'False alarm','Miss', 'Correct Rejection'])
            dFFPlot.ax[1, 0].set_title('Cue')
            dFFPlot.ax[1, 0].set_ylabel('dFF')

            dFFPlot.ax[1, 1].plot(self.interpT, ProbeLick_boot['bootAve'], color = cmap(0.25))
            dFFPlot.ax[1, 1].fill_between(self.interpT, ProbeLick_boot['bootLow'], ProbeLick_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.25))
            dFFPlot.ax[1, 1].plot(self.interpT, ProbeNoLick_boot['bootAve'], color = cmap(0.75))
            dFFPlot.ax[1, 1].fill_between(self.interpT, ProbeNoLick_boot['bootLow'], ProbeNoLick_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.75))
            dFFPlot.ax[1, 1].legend(['Probe lick', 'Probe no lick'])
            dFFPlot.ax[1, 1].set_title('Cue')
            dFFPlot.ax[1, 1].set_ylabel('dFF')

            dFFPlot.fig.set_size_inches(14, 10, forward=True)

            #plt.show()

            # save file
            dFFPlot.save_plot('cell' + str(cc) + '.tif', 'tif', savefigpath)
            plt.close()

            # plot  dFF with running?

    def plot_dFF_singleCell(self, cellID, trials):
        # plot the average dFF curve of a given cell for hit/FA/Hit/Miss trials

        # get the trials
        dFFCellPlot = StartPlots()

        dFF = self.dFF_aligned[:,:,cellID]
        Ind = 0
        for trial in trials:
            if self.beh['sound_num'][trial] in [1, 2, 3, 4]:
                c = (1, 0, 0)
            elif self.beh['sound_num'][trial] in [5, 6, 7, 8]:
                c = (0, 1, 0)
            else:
                c = (0, 0, 1)

            dFFCellPlot.ax.plot(self.interpT, dFF[:,trial]+Ind*1, color=c)
            Ind = Ind + 1

        plt.show()

    def process_X(self, regr_time, choiceList, rewardList, nTrials, nCells, trial):
        # re-arrange the behavior and dFF data for linear regression
        X = np.zeros((14,len(regr_time)))
        #
        Y = np.zeros((nCells, len(regr_time)))

        # need to determine whether to use exact frequency or go/no go/probe
        # separate into go/no go trials (for probe trials: 9-12: go; 13-16 no go
        # previous + next stimulus
        go_stim = [1,2,3,4,9,10,11,12]
        nogo_stim = [5,6,7,8,13,14,15,16]

        X[1, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial] in go_stim else 0]
        if trial == 0:
            X[2,:] = np.ones(len(regr_time)) * np.nan
        else:
            X[2, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial-1] in go_stim else 0]
        if trial == nTrials-1:
            X[0, :] = np.ones(len(regr_time)) * np.nan
        else:
            X[0, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial+1] in go_stim else 0]

        # choice: lick = 1; no lick = -1
        X[3, :] = np.ones(len(regr_time)) * (
            choiceList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[4, :] = np.ones(len(regr_time)) * (choiceList[trial])
        X[5, :] = np.ones(len(regr_time)) * (choiceList[trial - 1] if trial > 0 else np.nan)

        # reward
        X[6, :] = np.ones(len(regr_time)) * (
            rewardList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[7, :] = np.ones(len(regr_time)) * rewardList[trial]
        X[8, :] = np.ones(len(regr_time)) * (rewardList[trial - 1] if trial > 0 else np.nan)

        # interaction
        X[9, :] = X[3, :] * X[6, :]
        X[10, :] = X[4, :] * X[7, :]
        X[11, :] = X[5, :] * X[8, :]

        # running speed and licks
        tStep = np.nanmean(np.diff(regr_time))
        licks = self.beh['licks'][trial]

        for tt in range(len(regr_time)):
            t_start = regr_time[tt] - tStep / 2
            t_end = regr_time[tt] + tStep / 2
            temp_run = self.run_aligned[:, trial]
            X[12, tt] = np.nanmean(
                temp_run[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

            X[13, tt] = np.sum(np.logical_and(licks>=t_start+self.beh['onset'][trial],
                                              licks<t_end+self.beh['onset'][trial]))

            # dependent variable: dFF
            for cc in range(nCells):
                temp_dFF = self.dFF_aligned[:, trial, cc]
                Y[cc, tt] = np.nanmean(
                    temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return X, Y

    def linear_model(self, n_predictors):
        # arrange the independent variables and dependent variables for later linear regression
        # model:
        # y = b0 + b1*cue + b2* cn+1 + b3* cn + b4* cn-1 + b5* rn+1 + b6*rn + b7*rn-1 + b8* cn+1*rn+1 + b9* cn*rn + b10* cn-1*rn-1 + b11* running_speed

        tStart = -2.95
        tEnd = 4.95
        tStep = 0.1
        regr_time = np.arange(tStart, tEnd, tStep)
        nTrials = self.beh.shape[0]
        nCells = self.fluo.shape[1]-2
        independent_X = np.zeros((n_predictors, nTrials, len(regr_time)))
        dFF_Y = np.zeros((nCells, nTrials, len(regr_time)))

        choiceList = [1 if lick > 0 else 0 for lick in self.beh['licks_out']]
        rewardList = self.beh['reward']

        # parallel computing
        n_jobs = -1

        # Parallelize the loop over `trial`
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_X)(regr_time, choiceList, rewardList, nTrials, nCells, trial) for trial in tqdm(range(nTrials)))
        #dFF_Y = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_Y)(regr_time, nCells, trial) for trial in tqdm(range(nTrials)))

        # unpack the result of parallel computing
        for tt in range(nTrials):
            independent_X[:,tt,:], dFF_Y[:,tt,:] = results[tt]

        return np.array(independent_X), np.array(dFF_Y), regr_time

    def linear_regr(self, X, y, regr_time, saveDataPath):
        # try logistic regression for cues?
        """
        x： independent variables
        Y: dependent variables
        regr_t: time relative to cue onset
        output:
        coefficients, p-value
        R-square
        amount of variation explained
        """
        # linear regression model
        # cue, choice, reward, running speed,
        # cue (16 categorical variables)
        # choice: n-1, n, n+1 (n+1 should be a control since the choice depend on the cues)
        # reward: n-1, n, n+1 (n+1 control)
        # choice x reward： n-1, n, n+1
        # running speed

        # auto-correlation: run MLR first, then check residuals
        # reference: https://stats.stackexchange.com/questions/319296/model-for-regression-when-independent-variable-is-auto-correlated
        # fit the regression model
        nCells = self.fluo.shape[1] - 2
        n_predictors = X.shape[0]
        MLRResult = {'coeff': np.zeros((n_predictors, len(regr_time), nCells)), 'pval': np.zeros((n_predictors, len(regr_time), nCells)), 'r2': np.zeros((len(regr_time), nCells))}

        n_jobs = -1

        # Parallelize the loop over `trial`
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(self.run_MLR)(X[:,:,tt],y[:,:,tt]) for tt in
            tqdm(range(len(regr_time))))

        for tt in range(len(regr_time)):
            MLRResult['coeff'][:,tt,:], MLRResult['pval'][:,tt,:], MLRResult['r2'][tt,:] = results[tt]

        MLRResult['regr_time'] = regr_time

        # determine significant cells (for current stimulus, choice, outcome)
        # criteria: 3 consecutive significant time points, or 10 total significant time points
        sigCells = {}
        sigCells['stimulus'] = []
        sigCells['choice'] = []
        sigCells['outcome'] = []
        pThresh = 0.01
        for cc in range(nCells):
            stiPVal = MLRResult['pval'][1,:,cc]
            choicePVal = MLRResult['pval'][4,:,cc]
            outcomePVal = MLRResult['pval'][7,:,cc]
            if sum(stiPVal<0.01) >= 10 or count_consecutive(stiPVal<0.01)>=3:
                sigCells['stimulus'].append(cc)
            if sum(choicePVal < 0.01) >= 10 or count_consecutive(choicePVal < 0.01) >= 3:
                sigCells['choice'].append(cc)
            if sum(outcomePVal < 0.01) >= 10 or count_consecutive(outcomePVal < 0.01) >= 3:
                sigCells['outcome'].append(cc)

        MLRResult['sigCells'] = sigCells
        with open(saveDataPath, 'wb') as pf:
            pickle.dump(MLRResult, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()
        return MLRResult

    def plotMLRResult(self, MLRResult, labels, neuroRaw, saveFigPath):
        # get the average coefficient plot and fraction of significant neurons
        varList =labels
        # average coefficient
        nPredictors = MLRResult['coeff'].shape[0]

        coeffPlot = StartSubplots(4,4, ifSharey=True)

        maxY = 0
        minY = 0
        for n in range(nPredictors):
            tempBoot = bootstrap(MLRResult['coeff'][n,:,:],1, 1000)
            tempMax = max(tempBoot['bootHigh'])
            tempMin = min(tempBoot['bootLow'])
            if tempMax > maxY:
                maxY = tempMax
            if tempMin < minY:
                minY = tempMin
            coeffPlot.ax[n//4, n%4].plot(MLRResult['regr_time'], tempBoot['bootAve'], c =(0,0,0))
            coeffPlot.ax[n // 4, n % 4].fill_between(MLRResult['regr_time'], tempBoot['bootLow'], tempBoot['bootHigh'],
                                          alpha=0.2,  color = (0.7,0.7,0.7))
            coeffPlot.ax[n//4, n%4].set_title(varList[n])
        plt.ylim((minY,maxY))
        plt.show()
        coeffPlot.save_plot('Average coefficient.tif','tiff', saveFigPath)

        # fraction of significant neurons
        sigPlot = StartSubplots(4, 4, ifSharey=True)
        pThresh = 0.001
        nCell = MLRResult['coeff'].shape[2]

        # binomial test to determine signficance

        for n in range(nPredictors):
            fracSig = np.sum(MLRResult['pval'][n, :, :]<pThresh,1)/nCell
            pResults = [binomtest(x,nCell,p=pThresh).pvalue for x in np.sum(MLRResult['pval'][n, :, :]<pThresh,1)]
            sigPlot.ax[n // 4, n % 4].plot(MLRResult['regr_time'], fracSig, c=(0, 0, 0))
            sigPlot.ax[n // 4, n % 4].set_title(varList[n])

            if n//4 == 0:
                sigPlot.ax[n // 4, n % 4].set_ylabel('Fraction of sig')
            if n > 8:
                sigPlot.ax[n // 4, n % 4].set_xlabel('Time from cue (s)')
            # plot the signifcance bar
            dt = np.mean(np.diff(MLRResult['regr_time']))
            for tt in range(len(MLRResult['regr_time'])):
                if pResults[tt]<0.05:
                    sigPlot.ax[n//4, n%4].plot(MLRResult['regr_time'][tt]+dt*np.array([-0.5,0.5]), [0.5,0.5],color=(255/255, 189/255, 53/255), linewidth = 5)
        plt.ylim((0,0.6))
        plt.show()
        sigPlot.save_plot('Fraction of significant neurons.tif', 'tiff', saveFigPath)

        # plot r-square
        r2Boot = bootstrap(MLRResult['r2'], 1, 1000)
        r2Plot = StartPlots()
        r2Plot.ax.plot(MLRResult['regr_time'], r2Boot['bootAve'],c=(0, 0, 0))
        r2Plot.ax.fill_between(MLRResult['regr_time'], r2Boot['bootLow'], r2Boot['bootHigh'],
                                                 color=(0.7, 0.7, 0.7))
        r2Plot.ax.set_title('R-square')
        r2Plot.save_plot('R-square.tif', 'tiff', saveFigPath)


        """plot significant neurons"""
        sigCells = MLRResult['sigCells']
        cellstat = []
        for cell in range(neuronRaw.Fraw.shape[0]):
            if neuronRaw.cells[cell, 0] > 0:
                cellstat.append(neuronRaw.stat[cell])

        fluoCellPlot = StartPlots()
        im = np.zeros((256, 256,3))

        #for cell in range(decode_results[var]['importance'].shape[0]):
        for cell in range(len(cellstat)):
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
            if cell not in \
                    set(sigCells['choice'])|set(sigCells['outcome'])|set(sigCells['stimulus']):
                im[ys, xs] = [0.7, 0.7, 0.7]

        for cell in sigCells['choice']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys,xs] = [0,0,0]
            im[ys, xs] = np.add(im[ys, xs], [1.0, 0.0, 0.0])
        for cell in sigCells['outcome']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,1.0,0.0])
        for cell in sigCells['stimulus']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,0.0,1.0])
        action_patch = mpatches.Patch(color=(1,0,0), label='Action')
        outcome_patch = mpatches.Patch(color=(0,1,0), label = 'Outcome')
        stimulus_patch = mpatches.Patch(color=(0, 0, 1), label='Stimulus')
        # Create a custom legend with the green patch
        plt.legend(handles=[action_patch, outcome_patch, stimulus_patch],loc='center left',bbox_to_anchor=(1, 0.5))
        fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        plt.show()

        fluoCellPlot.save_plot('Regression neuron coordinates.tiff', 'tiff', saveFigPath)

    def run_MLR(self, x, y):
        # running individual MLR for parallel computing
        nCells = y.shape[0]
        n_predictors = x.shape[0]
        coeff = np.zeros((n_predictors, nCells))
        pval = np.zeros((n_predictors, nCells))
        rSquare = np.zeros((nCells))

        x = sm.add_constant(np.transpose(x))
        for cc in range(nCells):
            model = sm.OLS(y[cc,1:-1], x[1:-1,:]).fit()
            coeff[:,cc] = model.params[1:]
            pval[:,cc] = model.pvalues[1:]
            rSquare[cc] = model.rsquared

        return coeff, pval, rSquare

    def decoding_old(self, signal, decodeVar, varList, trialMask, classifier, regr_time):
        """
        function to decode behavior from neural activity
        running speed/reward/action/stimulus
        signal: neural signal. n x T
        var: variable to be decoded
        trialMask: trials to consider with specified conditions (Hit/FA etc.)
        """

        # run decoding for every time bin
        trialInd = np.arange(signal.shape[1])
        nullRepeats = 20
        decode_perform = {}
        #decode_perform['action'] = np.zeros(len(regr_time))
        #decode_perform['stimulus'] = np.zeros(len(regr_time))
        decode_null = {}
        #decode_null['action'] = np.zeros((len(regr_time), nullRepeats))
        #decode_null['stimulus'] = np.zeros((len(regr_time), nullRepeats))

        for varname in varList:
            decode_perform[varname] =  np.zeros(len(regr_time))
            decode_null[varname] = np.zeros((len(regr_time), nullRepeats))
            n_jobs = -1

        # Parallelize the loop over `trial`
            results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                delayed(self.run_decoder)(
                    signal, decodeVar,varname,trialInd,trialMask,
                    nullRepeats, classifier, idx) for idx in
                    tqdm(range(len(regr_time))))

            for tt in range(len(regr_time)):
                t1,t2=results[tt]
                decode_perform[varname][tt] = t1[varname]
                decode_null[varname][tt,:] = t2[varname]

        return decode_perform, decode_null

    # define the function for parallel computing
    def run_decoder_old(self, signal,decodeVar,varname,trialInd, trialMask,
                    nullRepeats, classifier, idx):
        # function for parallel computing
        if varname == 'action':
            data = {
                'raster': signal[:, trialMask, idx].transpose(),
                'action': decodeVar['action'][trialMask],
                #varname:decodeVar[varname][trialMask],
                #'stimulus': decodeVar['stimulus'][trialMask]
                'trial': trialInd[trialMask],
            }
            conditions = {
                'action': [1, 0],
            }
        elif varname == 'stimulus':
            data = {
                'raster': signal[:, trialMask, idx].transpose(),
                'stimulus': decodeVar['stimulus'][trialMask],
                'trial': trialInd[trialMask],
            }
            conditions = {
                'stimulus': {
                    'go': lambda d: d['stimulus'] <= 4,
                    'nogo': lambda d: (d['stimulus'] >= 5) & (d['stimulus'] <= 8)
                }  # this should be stimulus tested in the session
            }

        dec = Decodanda(
            data=data,
            conditions=conditions,
            classifier=classifier
        )

        performance, null = dec.decode(
            training_fraction=0.5,  # fraction of trials used for training
            cross_validations=10,  # number of cross validation folds
            nshuffles=nullRepeats)

        return performance, null

    def decoding(self, decodeSig, decodeVar, varList, trialMask, subTrialMask, classifier, regr_time, saveDataPath):

        # check if results already exist
        if os.path.exists(saveDataPath):
            print('Decoder exists, loading...')
            with open(saveDataPath, 'rb') as pf:
                # Load the data from the pickle file
                decode_results = pickle.load(pf)
                pf.close()
        else:
            decode_results = {}
            nCells = decodeSig.shape[0]
            n_shuffle = 20
            for varname in varList:

                decode_results[varname] = {}
                decode_results[varname]['classifier'] = []
                decode_results[varname]['ctrl'] = np.zeros((n_shuffle, len(regr_time)))
                decode_results[varname]['accuracy'] = np.zeros(len(regr_time))
                decode_results[varname]['importance'] = np.zeros((nCells,len(regr_time)))
                decode_results[varname]['params'] = {}
                decode_results[varname]['params']['n_estimators'] = np.zeros(len(regr_time))
                decode_results[varname]['params']['max_depth'] = np.zeros(len(regr_time))
                decode_results[varname]['params']['min_samples_leaf'] = np.zeros(len(regr_time))
                decode_results[varname]['prediction'] = {}
                decode_results[varname]['confidence'] = np.zeros((len(np.unique(decodeVar[varname])),
                                                                  len(regr_time)))
                n_jobs = -1


                # Parallelize the loop over `trial`
                results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                    delayed(self.run_decoder)(
                        decodeSig[:,:,idx].transpose(), decodeVar[varname],trialMask,
                        classifier, n_shuffle) for idx in
                    tqdm(range(len(regr_time))))

                #for tt in range(len(regr_time)):
                # unpacking results
                for rr in range(len(results)):
                    tempResults = results[rr]
                    decode_results[varname]['ctrl'][:,rr] = tempResults['ctrl']
                    decode_results[varname]['accuracy'][rr] = tempResults['accuracy']
                    decode_results[varname]['importance'][:,rr] = tempResults['importance']

                    decode_results[varname]['params']['n_estimators'][rr] = tempResults['params']['n_estimators']
                    decode_results[varname]['params']['max_depth'][rr] = tempResults['params']['max_depth']
                    decode_results[varname]['params']['min_samples_leaf'][rr] = tempResults['params']['min_samples_leaf']
                    decode_results[varname]['classifier'].append(tempResults['classifier'])
                    decode_results[varname]['confidence'][:,rr] = tempResults['confidence']
                # with trained model: decode action and cue in false alarm trials
                for key in subTrialMask.keys():
                    decode_results[varname]['prediction'][key] = np.zeros(len(regr_time))
                    # in probe trials, 2 -> 0; 3 -> 1

                    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                        delayed(self.run_decoder_trained_model)(
                        decode_results[varname]['classifier'][idx],
                        decodeSig[:,:,idx].transpose(),
                        decodeVar[varname],subTrialMask[key]) for idx in
                        tqdm(range(len(regr_time))))

                    for rr in range(len(results)):
                        decode_results[varname]['prediction'][key][rr] = results[rr]

            # save the decoding results
            for key in varList:
                del decode_results[key]['classifier']
            decode_results['time'] = regr_time
            decode_results['var'] = varList

            with open(saveDataPath, 'wb') as pf:
                pickle.dump(decode_results, pf, protocol=pickle.HIGHEST_PROTOCOL)
                pf.close()
            #return decode_perform, decode_null

        return decode_results

    def decode_analysis(self, neuronRaw, saveDataPath, saveFigPath):
        ## do some other analysis
        # plot decoding accuracy and control
        # plot decoding accuracy for false alarm trials
        #  identify cells with high importance, mark their location
        with open(saveDataPath, 'rb') as pf:
            # Load the data from the pickle file
            decode_results = pickle.load(pf)
            pf.close()

        # plot decoding accuracy for different variables
        decode_var = decode_results['var']
        nVars = len(decode_var)

        ''' plot decoding accuracy '''
        # check outcome
        decodePlot = StartSubplots(1, nVars, ifSharey=True)

        for n in range(nVars):
            decodePlot.ax[n].plot(decode_results['time'],
                                  decode_results[decode_var[n]]['accuracy'], c=(1, 0, 0))
            decodePlot.ax[n].set_title(decode_var[n])

            if n == 0:
                decodePlot.ax[n].set_ylabel('Decoding accuracy')
                decodePlot.ax[n].set_xlabel('Time from cue (s)')
            # plot null control
            ctrl_results = decode_results[decode_var[n]]['ctrl']
            bootCtrl = bootstrap(ctrl_results.transpose(),1, 0, n_sample=50)
            decodePlot.ax[n].fill_between(decode_results['time'], bootCtrl['bootLow'],
                                       bootCtrl['bootHigh'],color=(0.7, 0.7, 0.7))
            decodePlot.ax[n].plot(decode_results['time'], bootCtrl['bootAve'], c=(0,0,0))

        decodePlot.save_plot('Decoding accuracy.tiff', 'tiff', saveFigPath)

        '''plot decoding accuracy in hit/false alarm/correct rejection trials'''
        decodeFAPlot = StartSubplots(1, nVars, ifSharey=True)

        for n in range(nVars):
            plot_y1 = decode_results[decode_var[n]]['prediction']['Hit']
            plot_y2 = decode_results[decode_var[n]]['prediction']['FA']
            plot_y3 = decode_results[decode_var[n]]['prediction']['CorRej']
            decodeFAPlot.ax[n].plot(decode_results['time'],
                                 plot_y1, c=(1, 0, 0))
            decodeFAPlot.ax[n].plot(decode_results['time'],
                                    plot_y2, c=(0, 1, 0))
            decodeFAPlot.ax[n].plot(decode_results['time'],
                                    plot_y3, c=(0, 0, 1))
            decodeFAPlot.ax[n].legend(['Hit', 'FA', 'CorRej'])
            decodeFAPlot.ax[n].set_title(decode_var[n])

            if n == 0:
                decodeFAPlot.ax[n].set_ylabel('Decoding accuracy (Trial types)')
            decodeFAPlot.ax[n].set_xlabel('Time from cue (s)')
            # plot null control
            ctrl_results = decode_results[decode_var[n]]['ctrl']
            bootCtrl = bootstrap(ctrl_results.transpose(),1, 0, n_sample=50)
            decodeFAPlot.ax[n].fill_between(decode_results['time'], bootCtrl['bootLow'],
                                       bootCtrl['bootHigh'],color=(0.7, 0.7, 0.7))
            decodeFAPlot.ax[n].plot(decode_results['time'], bootCtrl['bootAve'], c=(0,0,0))

        decodeFAPlot.save_plot('Decoding accuracy (Trial types).tiff', 'tiff', saveFigPath)

        '''plot decoding accuracy in probe trials'''
        decodeProbePlot = StartSubplots(1, nVars, ifSharey=True)

        for n in range(nVars):
            plot_y = decode_results[decode_var[n]]['prediction']['probe']
            decodeProbePlot.ax[n].plot(decode_results['time'],
                                 plot_y, c=(1, 0, 0))
            decodeProbePlot.ax[n].set_title(decode_var[n])

            if n == 0:
                decodeProbePlot.ax[n].set_ylabel('Decoding accuracy (Probe)')
                decodeProbePlot.ax[n].set_xlabel('Time from cue (s)')
            # plot null control
            ctrl_results = decode_results[decode_var[n]]['ctrl']
            bootCtrl = bootstrap(ctrl_results.transpose(),1, 0, n_sample=50)
            decodeProbePlot.ax[n].fill_between(decode_results['time'], bootCtrl['bootLow'],
                                       bootCtrl['bootHigh'],color=(0.7, 0.7, 0.7))
            decodeProbePlot.ax[n].plot(decode_results['time'], bootCtrl['bootAve'], c=(0,0,0))

        decodeProbePlot.save_plot('Decoding accuracy (Probe).tiff', 'tiff', saveFigPath)

        ''' examine the important neurons'''
        # determine the relative importance by median importance between 0-3 s from cue
        importantNeurons = {}
        regr_time = decode_results['time']
        win = np.logical_and(regr_time > 0, regr_time < 3)
        p_thresh = 0.01
        for var in decode_var:
            # get the neurons that is significantly more than population median
            popMedianImportance = np.median(decode_results[var]['importance'][:,win])
            importantNeurons[var] = []
            for n in range(decode_results[var]['importance'].shape[0]):
                _, p_val = wilcoxon(decode_results[var]['importance'][n,win]-popMedianImportance)
                if p_val < p_thresh:
                    importantNeurons[var].append(n)

        # plot neuron position
        cellstat = []
        for cell in range(neuronRaw.Fraw.shape[0]):
            if neuronRaw.cells[cell, 0] > 0:
                cellstat.append(neuronRaw.stat[cell])

        fluoCellPlot = StartPlots()
        im = np.zeros((256, 256,3))

        #for cell in range(decode_results[var]['importance'].shape[0]):
        for cell in range(len(cellstat)):
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
            if cell not in \
                    set(importantNeurons['action'])|set( importantNeurons['outcome'])|set( importantNeurons['stimulus']):
                im[ys, xs] = [0.7, 0.7, 0.7]

        for cell in importantNeurons['action']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys,xs] = [0,0,0]
            im[ys, xs] = np.add(im[ys, xs], [1.0, 0.0, 0.0])
        for cell in importantNeurons['outcome']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,1.0,0.0])
        for cell in importantNeurons['stimulus']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,0.0,1.0])
        action_patch = mpatches.Patch(color=(1,0,0), label='Action')
        outcome_patch = mpatches.Patch(color=(0,1,0), label = 'Outcome')
        stimulus_patch = mpatches.Patch(color=(0, 0, 1), label='Stimulus')
        # Create a custom legend with the green patch
        plt.legend(handles=[action_patch, outcome_patch, stimulus_patch],loc='center left',bbox_to_anchor=(1, 0.5))
        fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        plt.show()

        fluoCellPlot.save_plot('Decoding neuron coordinates.tiff', 'tiff', saveFigPath)

    def run_decoder(self, input_x, input_y, trialMask, classifier, n_shuffle):
        # hand-made decoder
        # return a trained decoder that can be used to decode subset of signals in a manner of testing set


        x = input_x[trialMask,:]
        y = input_y[trialMask]
        # need to make a balanced training set
        X_test, y_test, X_train, y_train = self.get_train_test(x, y, test_size = 0.5, random_state = 66)

        #from sklearn.decomposition import PCA
        #x_transform = PCA(n_components=5).fit_transform(X_train)
        #plt.scatter(x_transform[y_train==0,2], x_transform[y_train==0,1], c=(1,0,0))
        #plt.scatter(x_transform[y_train == 1, 2], x_transform[y_train == 1, 1], c=(0, 1, 0))
        from sklearn import model_selection

        if classifier == 'RandomForest':

            rfc = RFC()
            n_estimators = [int(w) for w in np.linspace(start = 10, stop = 500, num=10)]
            max_depth = [int(w) for w in np.linspace(5, 20, num=10)] # from sqrt(n) - n/2
            min_samples_leaf = [0.1]
            max_depth.append(None)

            # create random grid
            random_grid = {
                'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'max_depth': max_depth
            }
            rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                                            n_iter=100, cv=3, verbose=False,
                                            random_state=42, n_jobs=-1)
            #rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
            #                                n_iter=100, cv=5, verbose=False,
            #                                random_state=42)

            # Fit the model
            rfc_random.fit(X_train, y_train)
            # print results
            #print(rfc_random.best_params_)

            best_n_estimators = rfc_random.best_params_['n_estimators']
            #best_n_estimators = 10
            #best_max_features = rfc_random.best_params_['max_features']
            best_max_depth = rfc_random.best_params_['max_depth']
            best_min_samples_leaf = rfc_random.best_params_['min_samples_leaf']
            model = RFC(n_estimators = best_n_estimators,
                           max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf, class_weight='balanced')

            # get control value by shuffling trials

            model.fit(X_train, y_train)

            # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
            #from sklearn.metrics import balanced_accuracy_score
            # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
            # calculate decoding accuracy based on confusion matrix
            best_predict = model.predict(X_test)
            proba_estimates = model.predict_proba(X_test)
            pred = confusion_matrix(y_test, best_predict)
            pred_accuracy = np.trace(pred)/np.sum(pred)

            # feature importance
            importance = model.feature_importances_
            # need to return: classfier (to decode specific trial type later)
            #                 shuffled accuracy (control)
            #                 accuracy (decoding results)
            #                 importance
            #                 best parameters of the randomforest decoder


            # control
            pred_shuffle = np.zeros(n_shuffle)
            for ii in range(n_shuffle):
                xInd = np.arange(len(y_test))
                X_test_shuffle = np.zeros((X_test.shape))
                for cc in range(X_train.shape[1]):
                    np.random.shuffle(xInd)
                    X_test_shuffle[:,cc] = X_test[xInd,cc]

                predict_shuffle = model.predict(X_test_shuffle)
                pred = confusion_matrix(y_test, predict_shuffle)
                pred_shuffle[ii] = np.trace(pred) / np.sum(pred)

        elif classifier == 'SVC':
            model = SVC(kernel='linear')

            # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
            # from sklearn.metrics import balanced_accuracy_score
            # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
            # calculate decoding accuracy based on confusion matrix
            model.fit(X_train,y_train)
            best_predict = model.predict(X_test)
            pred = confusion_matrix(y_test, best_predict)
            pred_accuracy = np.trace(pred) / np.sum(pred)

            # feature importance
            importance = model.coef_
            pred_shuffle = np.zeros(n_shuffle)
            for ii in range(n_shuffle):
                xInd = np.arange(len(y_test))
                X_test_shuffle = np.zeros((X_test.shape))
                for cc in range(X_train.shape[1]):
                    np.random.shuffle(xInd)
                    X_test_shuffle[:, cc] = X_test[xInd, cc]

                predict_shuffle = model.predict(X_test_shuffle)
                pred = confusion_matrix(y_test, predict_shuffle)
                pred_shuffle[ii] = np.trace(pred) / np.sum(pred)


        decoder = {}
        decoder['classifier'] = model
        decoder['ctrl'] = pred_shuffle
        decoder['accuracy'] = pred_accuracy
        decoder['importance'] = importance
        if classifier == 'RandomForest':
            decoder['params'] = rfc_random.best_params_
            decoder['confidence'] = np.mean(proba_estimates,0)
        return decoder

    def run_decoder_trained_model(self, decoder, input_x, input_y, subTrialMask):
        # decode subset of trials with already trained model

        x = input_x[subTrialMask,:]
        y = input_y[subTrialMask]


        # calculate decoding accuracy based on confusion matrix
        predict = decoder.predict(x)

        pred_accuracy = np.sum(predict==y)/len(y)

        return pred_accuracy

    def get_train_test(self, X, y, test_size, random_state):
        # check number of classes
        random.seed(random_state)
        classes = np.unique(y)
        nClass = len(np.unique(y))

        instance_class = np.zeros(nClass)
        for cc in range(nClass):
            instance_class[cc] = np.sum(y==classes[cc])

        minIns = np.min(instance_class)
        minInd = np.unravel_index(np.argmin(instance_class),instance_class.shape)
        minClass = classes[minInd]

        # split the trials based on test_size and the class with minimum instances
        classCountTest = np.sum(y==minClass)*test_size
        trainInd = []
        testInd = []
        for nn in range(nClass):
            tempClassInd = np.arange(len(y))[y==classes[nn]]
            tempTestInd = random.choices(tempClassInd,
                                          k=int(classCountTest))
            IndRemain = np.setdiff1d(tempClassInd,tempTestInd)
            tempTrainInd = random.choices(IndRemain,
                                          k=int(classCountTest))
            testInd = np.concatenate([testInd,tempTestInd])
            trainInd = np.concatenate([trainInd,tempTrainInd])

        trainInd = trainInd.astype(int)
        testInd = testInd.astype(int)

        X_train = X[trainInd,:]
        X_test = X[testInd,:]
        y_train = y[trainInd]
        y_test = y[testInd]

        return X_train,y_train, X_test, y_test


    def clusering(self):
        # hierachical clustering
        # cluster pure neural activity? cluster average neural activity in different trials?
        # cluster correlation coefficient?
        pass

if __name__ == "__main__":
    animal, session = 'JUV015', '220409'

    beh_folder = "C:\\Users\\linda\\Documents\\GitHub\\madeline_go_nogo\\data"
    beh_file = "JUV015-220409-behaviorLOG.mat"
    trialbytrial = GoNogoBehaviorMat(animal, session, os.path.join(beh_folder, beh_file))

    # read the raw fluorescent data from suite2P
    input_folder = r"C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\suite2p_output"
    gn_series = Suite2pSeries(input_folder)

    # align behavior with fluorescent data
    gn_series.realign_time(trialbytrial)

    # save file
    fluo_file = r'C:\Users\linda\Documents\GitHub\madeline_imagingData\JUV015_220409_dff_df_file.csv'
    gn_series.neural_df.to_csv(fluo_file)

    # plot the cell location
    gn_series.plot_cell_location_dFF(np.arange(gn_series.neural_df.shape[1]-1))

    # dff_df = gn_series.calculate_dff(melt=False)

    beh_file = r'C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\behAnalysis.pickle'

    # build the linear regression model
    analysis = fluoAnalysis(beh_file,fluo_file)
    analysis.align_fluo_beh()

    # individual cell plots
    #trials = np.arange(20,50)
    #analysis.plot_dFF_singleCell(157, trials)
    # cell plots
    # analysis.plot_dFF(os.path.join(fluofigpath,'cells-combined-cue'))

    # build multiple linear regression
    # arrange the independent variables
    saveFigPath = r'C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\fluo_plot'

    n_predictors = 14
    labels = ['s(n+1)','s(n)', 's(n-1)','c(n+1)', 'c(n)', 'c(n-1)',
              'r(n+1)', 'r(n)', 'r(n-1)', 'x(n+1)', 'x(n)', 'x(n-1)', 'speed', 'lick']
    X, y, regr_time = analysis.linear_model(n_predictors)
    saveDataPath = r'C:\\Users\\linda\\Documents\\GitHub\\madeline_go_nogo\\data\\MLR.pickle'
    MLRResult = analysis.linear_regr(X[:,1:-2,:], y[:,1:-2,:], regr_time, saveDataPath)
    analysis.plotMLRResult(MLRResult, labels, saveFigPath)

    # for decoding
    # decode for action/outcome/stimulus
    decodeVar = {}

    decodeVar['action'] = X[4,:,0]
    decodeVar['outcome'] = X[7,:,0]
    decodeVar['stimulus'] = np.array([np.int(analysis.beh['sound_num'][x]) for x in range(len(analysis.beh['sound_num']))])
    decodeSig = y

    trialMask = decodeVar['stimulus'] <= 8
    # check false alarm trials, and probe trials
    subTrialMask = {}
    subTrialMask['FA'] = analysis.beh['trialType'] == -1
    subTrialMask['probe'] = decodeVar['stimulus'] > 8
    subTrialMask['Hit'] = analysis.beh['trialType'] == 2
    subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
    # stimulus 1-4: 1
    # stimulus 5-8: 0
    # stimulus 9-12；2
    # stimulus 13-16: 3
    tempSti = np.zeros(len(decodeVar['stimulus']))
    for ss in range(len(decodeVar['stimulus'])):
        if decodeVar['stimulus'][ss] <= 4:
            tempSti[ss] = 1
        elif decodeVar['stimulus'][ss] > 4 and decodeVar['stimulus'][ss] <= 8:
            tempSti[ss] = 0
        elif decodeVar['stimulus'][ss] > 8 and decodeVar['stimulus'][ss] <= 12:
            tempSti[ss] = 1
        elif decodeVar['stimulus'][ss] > 12:
            tempSti[ss] = 0
    decodeVar['stimulus'] = tempSti

    classifier = "RandomForest"
    varList = ['action','outcome','stimulus']
    saveDataPath = r'C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\decode.pickle'

    #decode_results = analysis.decoding(decodeSig, decodeVar, varList, trialMask,subTrialMask, classifier, regr_time, saveDataPath)
    neuronRaw = gn_series


    # train decode model on whole dataset, use it to decode FA trials later

    # conditions to decode:
    # error trials
    # neuron position
    # redundancy

    '''demixed PCA'''
    stim = np.array([np.int(analysis.beh['sound_num'][x]) for x in range(len(analysis.beh['sound_num']))])
    tempStim = np.zeros(len(stim))
    for ss in range(len(stim)):
        if stim[ss] <= 4:
            tempStim[ss] = 1
        elif stim[ss] > 4 and stim[ss] <= 8:
            tempStim[ss] = 0
        elif stim[ss] > 8 and stim[ss] <= 12:
            tempStim[ss] = 2
        elif stim[ss] > 12:
            tempStim[ss] = 3
    pcaVar = {'stim':tempStim, 'action':X[4,:,0]}
    x = 1
