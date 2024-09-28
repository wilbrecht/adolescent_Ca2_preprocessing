from abc import abstractmethod

import numpy as np
import pandas as pd
import h5py
import os
from pyPlotHW import StartPlots, StartSubplots
from utility_HW import bootstrap
import glob
from tqdm import tqdm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import minimize
from scipy.stats import norm, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time
from statsmodels.stats.multicomp import MultiComparison


import pandas as pd
import pickle

from behavior_base import PSENode, EventNode

# import matlab engine
import matlab.engine
eng = matlab.engine.start_matlab()
# From matlab file get_Headfix_GoNo_EventTimes.m:
# % eventID=1:    2P imaging frame TTL high
# % eventID=2:    2P imaging frame TTL low
# % eventID=3:    left lick in
# % eventID=4:    left lick out
# % eventID=44:   Last left lick out
# % eventID=5:    right lick 1n
# % eventID=6:    right lick out
# % eventID=66:   Last right lick out
# % eventID=7.01:  new trial, Sound 1 ON
# % eventID=7.02:  new trial, Sound 2 ON
# % eventID=7.0n:  new trial, Sound n ON
# % eventID=7.16:  new trial, Sound 16 ON
# % eventID=81.01: Correct No-Go (no-lick), unrewarded outcome
# % eventID=81.02: Correct Go (lick), unrewarded outcome
# % eventID=81.12: Correct Go (lick), 1 drop rewarded after direct delivery
# % eventID=81.22: Correct Go (lick), 2 drops rewarded (valve on)
# % eventID=82:02  False Go (lick), white noise on
# % eventID=83:    Missed to respond
# % eventID=84:    Aborted outcome
# % eventID=9.01:  Water Valve on 1 time (1 reward)
# % eventID=9.02:  Water Valve on 2 times (2 rewards)
# % eventID=9.03:  Water Valve on 3 times (3 rewards)

class BehaviorMat:
    code_map = {}
    fields = []  # maybe divide to event, ev_features, trial_features
    time_unit = None
    eventlist = None

    def __init__(self, animal, session):
        self.animal = animal
        self.session = session
        self.time_aligner = lambda s: s  # provides method to align timestamps

    @abstractmethod
    def todf(self):
        return NotImplemented

    def align_ts2behavior(self, timestamps):
        return self.time_aligner(timestamps)


class GoNogoBehaviorMat(BehaviorMat):
    code_map = {
        3: ('in', 'in'),
        4: ('out', 'out'),
        5: ('in', 'in'),    # not sure why some files contain 5 and 6 (for right lick)
        6: ('out', 'out'),
        44: ('out', 'out'),
        66: ('out', 'out'), # last right lick out
        81.01: ('outcome', 'no-go_correct_unrewarded'),
        81.02: ('outcome', 'go_correct_unrewarded'),
        81.12: ('outcome', 'go_correct_reward1'),
        81.22: ('outcome', 'go_correct_reward2'),
        82.02: ('outcome', 'no-go_incorrect'),
        82.01: ('outcome','no-go_correct_reward'),  # what is this?
        83: ('outcome', 'missed'),
        84: ('outcome', 'abort'),
        9.01: ('water_valve', '1'),
        9.02: ('water_valve', '2'),
        9.03: ('water_valve', '3')
    }
    # 7.0n -> sound n on (n = 1-16)
    for i in range(1, 17):
        code_map[(700 + i) / 100] = ('sound_on', str(i))

    fields = ['onset', 'first_lick_in', 'last_lick_out', 'water_valve_on', 'outcome', 'licks','running_speed','time_0']

    time_unit = 's'

    def __init__(self, animal, session, hfile):
        super().__init__(animal, session)
        self.hfile = h5py.File(hfile, 'r')
        self.animal = animal
        self.session = session
        self.trialN = len(self.hfile['out/result'])
        self.eventlist, self.runningSpeed = self.initialize_node()

        # get time_0 (normalize the behavior start time to t=0)
        for node in self.eventlist:
            # New tone signifies a new trial
            if node.event == 'sound_on':
                # get the time of cue onset in trial 1, normalize all following trials
                if node.trial_index() == 1:
                    self.time_0 = node.etime
                    break

        # get the timestamp for aligning with fluorescent data
        if isinstance(hfile, str):
            with h5py.File(hfile, 'r') as hf:
                if 'out/frame_time' in hf:
                    frame_time = np.array(hf['out/frame_time']).ravel()
                else:
                    frame_time = np.nan
        else:
            frame_time = np.array(hfile['out/frame_time']).ravel()
        self.time_aligner = lambda t: frame_time

        # a dictionary to save all needed behavior metrics
        self.saveData = dict()

    def initialize_node(self):
        code_map = self.code_map
        eventlist = EventNode(None, None, None, None)
        trial_events = np.array(self.hfile['out/GoNG_EventTimes'])
        if 'out/run_speed' in self.hfile:
            running_speed = np.array(self.hfile['out/run_speed'])
        else:
            running_speed = [np.nan]

        exp_complexity = None
        struct_complexity = None
        prev_node = None
        duplicate_events = [81.02, 81.12, 81.22, 82.02, 82.01]
        for i in range(len(trial_events)):
            eventID, eventTime, trial = trial_events[i]
            # check duplicate timestamps
            if prev_node is not None:
                if prev_node.etime == eventTime:
                    if eventID == prev_node.ecode:
                        continue
                    elif eventID not in duplicate_events:
                        print(f"Warning! Duplicate timestamps({prev_node.ecode}, {eventID})" +
                              f"at time {eventTime} in {str(self)}")
                elif eventID in duplicate_events:
                    print(f"Unexpected non-duplicate for {trial}, {code_map[eventID]}, {self.animal}, "
                          f"{self.session}")
            cnode = EventNode(code_map[eventID][0], eventTime, trial, eventID)
            eventlist.append(cnode)
            prev_node = cnode

        return eventlist, running_speed

    def to_df(self):
        columns = ['trial'] + self.fields
        result_df = pd.DataFrame(np.full((self.trialN, len(columns)), np.nan), columns=columns)
        result_df['animal'] = self.animal
        result_df['session'] = self.session
        result_df['time_0'] = 0 # update later
        result_df = result_df[['animal', 'session', 'trial'] + self.fields]

        result_df['trial'] = np.arange(1, self.trialN + 1)
        result_df['sound_num'] = pd.Categorical([""] * self.trialN, np.arange(1, 16 + 1), ordered=False)
        result_df['reward'] = pd.Categorical([""] * self.trialN, [-1, 0, 1], ordered=False)
        result_df['go_nogo'] = pd.Categorical([""] * self.trialN, ['go', 'nogo'], ordered=False)
        result_df['licks_out'] = np.full((self.trialN, 1), 0)
        result_df['quality'] = pd.Categorical(["normal"] * self.trialN, ['missed', 'abort', 'normal'], ordered=False)
        result_df['water_valve_amt'] = pd.Categorical([""] * self.trialN, [1, 2, 3], ordered=False)


        # add another entry to record all the licks
        result_df['licks'] = [[] for _ in range(self.trialN)] # convert to np.array later
        result_df['choice'] = pd.Categorical([""] * self.trialN, [0, 1], ordered=False)

        result_df['trialType'] = pd.Categorical([""] * self.trialN, [-4, -3, -2, -1, 0, 1, 2], ordered=False)

        for node in self.eventlist:
            # New tone signifies a new trial
            # trial should start with 'sound_on'
            if node.event == 'sound_on':
                # get the time of cue onset in trial 1, normalize all following trials
                if node.trial_index() == 1:
                    time_0 = node.etime
                    result_df['time_0'] = time_0

                result_df.at[node.trial_index()-1, 'onset'] = node.etime-time_0
                result_df.at[node.trial_index()-1, 'sound_num'] = int(self.code_map[node.ecode][1])


            elif node.event == 'in':
                # add a list contain all licks in the trial
                if node.trial_index() > 0:
                    if not result_df.at[node.trial_index()-1, 'licks']:
                        result_df.at[node.trial_index()-1, 'licks'] = [node.etime-time_0]
                    else:
                        result_df.at[node.trial_index()-1, 'licks'].append(node.etime-time_0)

                    if np.isnan(result_df.loc[node.trial_index()-1, 'first_lick_in']):
                        result_df.loc[node.trial_index()-1, 'first_lick_in'] = node.etime-time_0
            elif node.event == 'out':
                if node.trial_index() > 0:
                    result_df.loc[node.trial_index()-1, 'last_lick_out'] = node.etime-time_0
                    result_df.loc[node.trial_index()-1, 'licks_out'] += 1
            elif node.event == 'outcome':
                result_df.loc[node.trial_index()-1, 'outcome'] = node.etime-time_0
                outcome = self.code_map[node.ecode][1]
                # quality
                if outcome in ['missed', 'abort']:
                    result_df.loc[node.trial_index()-1, 'quality'] = outcome
                # reward
                # if '_correct_' in outcome:
                #     reward = int(outcome[-1]) if outcome[-1].isnumeric() else 0
                #     result_df.loc[node.trial_index()-1, 'reward'] = reward
                # else:
                #     result_df.loc[node.trial_index()-1, 'reward'] = -1
                # go nogo
                if outcome.startswith('go') or outcome == 'missed':
                    result_df.loc[node.trial_index()-1, 'go_nogo'] = 'go'
                elif outcome.startswith('no-go'):
                    result_df.loc[node.trial_index()-1, 'go_nogo'] = 'nogo'
            elif node.event == 'water_valve':
                num_reward = self.code_map[node.ecode][1]
                result_df.loc[node.trial_index()-1, 'water_valve_amt'] = int(num_reward)
                result_df.loc[node.trial_index()-1, 'water_valve_on'] = node.etime-time_0

        # align running speed to trials
        if len(self.runningSpeed)>1:   # make sure there is running data
            self.runningSpeed[:, 0] = self.runningSpeed[:, 0] - time_0

            result_df['running_speed'] = [[] for _ in range(self.trialN)]
            result_df['running_time'] = [[] for _ in range(self.trialN)]

        # remap trialType (eaiser for grouping trials based on cue and outcome）
        # -4: probeSti, no lick;
        # -3: probSti, lick;
        # -2: miss;
        # -1: false alarm;
        # 0: correct reject;
        # 1/2: hit for reward amount

        for tt in range(self.trialN):
            t_start = result_df.onset[tt]
            if len(self.runningSpeed)>1:
                if tt<self.trialN-1:
                    t_end = result_df.onset[tt+1]
                    result_df.at[tt, 'running_speed'] = self.runningSpeed[np.logical_and(self.runningSpeed[:,0]>=t_start-3, self.runningSpeed[:,0]<t_end),1].tolist()
                    result_df.at[tt, 'running_time'] = self.runningSpeed[
                        np.logical_and(self.runningSpeed[:, 0] >= t_start - 3, self.runningSpeed[:, 0] < t_end), 0].tolist()
                elif tt == self.trialN-1:
                    result_df.at[tt, 'running_speed'] = self.runningSpeed[self.runningSpeed[:, 0] >= t_start-3, 1].tolist()
                    result_df.at[tt, 'running_time'] = self.runningSpeed[self.runningSpeed[:, 0] >= t_start - 3, 0].tolist()
            # remap choice and reward
            # choice: 1/0 lick/no lick
            # reward: 1/0/-1 hit/correct rejection/false alarm,miss
            result_df.at[tt, 'choice'] = 0 if result_df.at[tt, 'licks_out'] == 0 else 1
            if result_df.sound_num[tt] in [9, 10, 11, 12, 13, 14, 15, 16]:
                result_df.at[tt, 'reward'] = 0
            elif result_df.sound_num[tt] in [1, 2, 3, 4]:
                result_df.at[tt, 'reward'] = -1 if result_df.at[tt, 'licks_out']==0 else 1
            elif result_df.sound_num[tt] in [5, 6, 7, 8]:
                result_df.at[tt, 'reward'] = 0 if result_df.at[tt, 'licks_out']==0 else -1

            if result_df.sound_num[tt] in [9, 10, 11, 12, 13, 14, 15, 16]:
                result_df.at[tt, 'trialType'] = -4 if result_df.at[tt,'licks_out'] == 0 else -3
            elif result_df.sound_num[tt] in [1, 2, 3, 4]:
                result_df.at[tt, 'trialType'] = -2 if result_df.at[tt, 'licks_out'] == 0 else 2
            elif result_df.sound_num[tt] in [5, 6, 7, 8]:
                result_df.at[tt, 'trialType'] = 0 if result_df.at[tt, 'licks_out']==0 else -1

        # save the data into self
        self.DF = result_df
        self.saveData['behFull'] = result_df

        return result_df

    def beh_cut(self, save_path):
        # obslete
        # animals stop to engage in the task in some sessions
        # should only apply in later session???
        # calculate the running d-prime and detect change point, then delete the
        # following trials

        # calculate running d-prime in 50 trial blocks
        nStep = 50
        nTrials = self.DF.shape[0]
        runningDprime = np.zeros(nTrials-nStep+1)
        runningHitRate = np.zeros(nTrials-nStep+1)
        for idx in range(nTrials-nStep+1):
            # use loglinear to calculate the d-prime
            # reference: Macmillan & Kaplan, 1985
            # adjusted_hitRate = (nHit+ nGo/nSum)/(nGo+1)
            # adjusted_FARate = (nFA+nNoGo/nSum)/(nNoGo+1)

            nTrialGo = np.sum(np.logical_or(self.DF['trialType'][idx:idx+nStep] == 2,
                                self.DF['trialType'][idx:idx+nStep] == -2))
            nTrialNoGo = np.sum(np.logical_or(self.DF['trialType'][idx:idx+nStep] == -1,
                                self.DF['trialType'][idx:idx+nStep] == 0))
            Hit_rate = (np.sum(self.DF['trialType'][idx:idx+nStep] == 2)+
                        nTrialGo/(nTrialGo+nTrialNoGo)) / (nTrialGo+1)
            FA_rate = (np.sum(self.DF['trialType'][idx:idx+nStep]== -1)+
                        nTrialNoGo/(nTrialNoGo+nTrialGo))/ (nTrialNoGo+1)

            runningHitRate[idx] = Hit_rate
            runningDprime[idx] = norm.ppf(Hit_rate) - norm.ppf(FA_rate)

        # change point detection using matlab function ischange
        try:
            TF, S1, S2 = eng.ischange(runningHitRate, 'linear', 'MaxNumChanges', 2, nargout=3)
        except:
            TF = [[np.nan]]
            S1 = [[np.nan]]
            S2 = [[np.nan]]

        if not np.isnan(TF[0][0]):
            hitThresh = 0.8
            segline = np.array(S1[0])*(np.arange(len(runningHitRate))) + np.array(S2[0])

            # find the point where runninghitrate is below 0.8
            xAxis = np.arange(len(runningHitRate))
            IndAbove = segline[0]<hitThresh

            cross_point = eng.ischange(np.double(IndAbove))[0]
            TF_hitThresh = [xAxis[i] for i in range(len(xAxis)) if cross_point[i]] # including change points both go below 1 and go above 1
            TF_belowThresh = []
            for k in range(len(TF_hitThresh)):
                if segline[0][TF_hitThresh[k] - 1] > hitThresh:
                    TF_belowThresh.append(TF_hitThresh[k])

            tfPoint = [xAxis[i] for i in range(len(xAxis)) if TF[0][i]]
            self.cutoff = 0
            self.ifCut = False
            # chech if TF_belowThresh is None

            if TF_belowThresh is not None:
                for ii in range(len(TF_belowThresh)):
            # check the point 1 by 1
                    if TF_belowThresh[ii] > 150: # critierion 1
            # finding the change point before I(ii)
                        if TF_belowThresh[ii] in tfPoint: # the point itself is a change point
                            if segline[0][TF_belowThresh[ii]] < hitThresh: # criterion 2
                                if np.max(segline[0][TF_belowThresh[ii] + 1:-1]) < segline[0][TF_belowThresh[ii] - 1]: # criterion 3
                                    self.cutoff = TF_belowThresh[ii]
                                    self.ifCut = True
                                    break
                        else:
                        # find the last trial when hit rate drop below threshold
                            tf_all = np.where(tfPoint < TF_belowThresh[ii])[0]
                            if tf_all.size == 0:
                                tf = 0
                            else:
                                tf = np.where(tfPoint < TF_belowThresh[ii])[0][-1]
                            tfP = tfPoint[tf]
                            if np.max(segline[0][TF_belowThresh[ii] + 1: -1]) < np.max([segline[0][tfP - 1], segline[0][tfP]]):
                    # criterion 3
                                self.ifCut = True
                                self.cutoff = TF_belowThresh[ii]
                                break

        else:
            self.cutoff = 0
            self.ifCut = False
            segline = np.zeros(len(runningHitRate))


        # plot the cut result
        cut_plot = StartPlots()
        cut_plot.ax.plot(runningDprime)
        cut_plot.ax.plot(runningHitRate)
        cut_plot.ax.plot(segline[0])
        if self.ifCut:
            cut_plot.ax.scatter(self.cutoff, runningHitRate[self.cutoff], s= 80, c='red')
        cut_plot.save_plot('Cut point.png', 'png', save_path)

        # save the result
        if self.ifCut:
            self.DFFull = self.DF
            self.DF = self.DF.iloc[0:self.cutoff]
            self.trialNFull = self.trialN
            self.trialN = self.cutoff

        self.saveData['ifCut'] = self.ifCut
        self.saveData['cutoff'] = self.cutoff
        self.saveData['behDF'] = self.DF
    def output_df(self, outfile, file_type='csv'):
        """
        saves the output of to_df() as a file of the specified type
        """
        if file_type == 'csv':
            self.DF.to_csv(outfile + '.csv')

    ### plot methods for behavior data
    # should add outputs later, for summary plots

    def beh_session(self, save_path, ifrun):

        # check if plot already exist
        plotname = os.path.join(save_path,'Behavior summary.svg')
        if not os.path.exists(plotname) or ifrun:
            # plot the outcome according to trials


            beh_plots = StartPlots()
            # hit trials
            if self.ifCut:
                behDF = self.DFFull
                trialNum = np.arange(self.trialNFull)
            else:
                behDF = self.DF
                trialNum = np.arange(self.trialN)

            beh_plots.ax.scatter(trialNum[behDF.trialType == 2], np.array(behDF.trialType[behDF.trialType == 2]),
                       s=100, marker='o')

            # miss trials
            beh_plots.ax.scatter(trialNum[behDF.trialType == -2], behDF.trialType[behDF.trialType == -2], s=100,
                       marker='x')

            # false alarm
            beh_plots.ax.scatter(trialNum[behDF.trialType == -1], behDF.trialType[behDF.trialType == -1], s=100,
                       marker='*')

            # correct rejection
            beh_plots.ax.scatter(trialNum[behDF.trialType == 0], behDF.trialType[behDF.trialType == 0], s=100,
                       marker='.')

            # probe lick
            beh_plots.ax.scatter(trialNum[behDF.trialType == -3], behDF.trialType[behDF.trialType == -3], s=100,
                       marker='v')

            # probe no lick
            beh_plots.ax.scatter(trialNum[behDF.trialType == -4], behDF.trialType[behDF.trialType == -4], s=100,
                       marker='^')

            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            beh_plots.ax.set_title('Session summary')
            beh_plots.ax.set_xlabel('Trials')
            beh_plots.ax.set_ylabel('Outcome')
            leg = beh_plots.legend(['Hit', 'Miss', 'False alarm', 'Correct rejection', 'Probe lick', 'Probe miss'])

            #legend.get_frame().set_linewidth(0.0)
            #legend.get_frame().set_facecolor('none')
            beh_plots.fig.set_figwidth(40)
            plt.show()

            # save the plot
            beh_plots.save_plot('Behavior summary.svg', 'svg', save_path)
            beh_plots.save_plot('Behavior summary.tif', 'tif', save_path)
        # trialbytrial.beh_session()

    def d_prime(self):
        # calculate d prime
        nTrialGo = np.sum(np.logical_or(self.DF['trialType']==2, self.DF['trialType']==-2))
        nTrialNoGo = np.sum(np.logical_or(self.DF['trialType'] == -1, self.DF['trialType'] == 0))
        Hit_rate = np.sum(self.DF['trialType'] == 2) / nTrialGo
        FA_rate = np.sum(self.DF['trialType'] == -1) / nTrialNoGo


        d_prime = norm.ppf(self.check_rate(Hit_rate)) - norm.ppf(self.check_rate(FA_rate))

        self.saveData['d-prime'] = d_prime

    def check_rate(self, rate):
        # for d-prime calculation
        # if value == 1, change to 0.9999
        # if value == 0, change to 0.0001
        if rate == 1:
            rate = 0.9999
        elif rate == 0:
            rate = 0.0001

        return rate

    def psycho_curve(self, save_path, ifrun):
        # get variables
        # hfile['out']['sound_freq'][0:-1]
        # use logistic regression
        # L(P(go)/(1-P(go)) = beta0 + beta_Go*S_Go + beta_NoGo * S_NoGo
        # reference: Breton-Provencher, 2022
        plotname = os.path.join(save_path,'psychometric.svg')
        if not os.path.exists(plotname) or ifrun:
            numSound = 16

            goCueInd = np.arange(1, 5)
            nogoCueInd = np.arange(5, 9)
            probeCueInd = np.arange(9, 17)

            goFreq = np.array([6.49, 7.07, 8.46, 9.17])
            nogoFreq = np.array([10.9, 11.9, 14.14, 15.41])
            probeFreq = np.array([6.77, 7.73, 8.81, 9.71, 10.29, 11.38, 12.97, 14.76])
            midFreq = (9.17+10.9)/2

            # %%
            # psychometric curve
            sound = np.arange(1, numSound + 1)
            numGo = np.zeros(numSound)

            # sort sound, base on the frequency
            soundIndTotal = np.concatenate((goCueInd, nogoCueInd, probeCueInd))
            soundFreqTotal = np.concatenate((goFreq, nogoFreq, probeFreq))

            sortedInd = np.argsort(soundFreqTotal)
            sortedIndTotal = soundIndTotal[sortedInd]
            sortedFreqTotal = soundFreqTotal[sortedInd]
            stiSortedInd = np.where(np.in1d(sortedIndTotal, np.concatenate((goCueInd, nogoCueInd))))[0]
            probeSortedInd = np.where(np.in1d(sortedIndTotal, probeCueInd))[0]

            for ss in range(len(numGo)):
                numGo[ss] = np.sum(np.logical_and(self.DF.sound_num == ss + 1,
                    self.DF.choice == 1))

                sound[ss] = np.sum(self.DF.sound_num == ss+1)

            sortednumGo = numGo[sortedInd]
            sortednumSound = sound[sortedInd]

            # save the data and frequency
            self.saveData['psycho_data'] = sortednumGo/sortednumSound
            self.saveData['psycho-sti'] = soundFreqTotal[sortedInd]

            # fit logistic regression
            y = sortednumGo[stiSortedInd] / sortednumSound[stiSortedInd]

            # replace 1s to 0.99999 to avoid perfect separation
            y[y==1] = 0.99999
            y[y==0] = 0.00001

            x = np.array((sortedFreqTotal[stiSortedInd],sortedFreqTotal[stiSortedInd]))
            x[0, 4:] = 0 # S_Go
            x[1, 0:4] = 0 # S_NoGo
            x = x.transpose()
            x = sm.add_constant(x)

            if np.count_nonzero(~np.isnan(y)) > 2:


                # drop nan values
                keepInd = np.logical_not(np.isnan(y))
                y = y[keepInd]
                x = x[keepInd,:]

                model = sm.Logit(y, x).fit(disp=False)

                # save the data
                self.saveData['L-fit'] = model.params

                # generating x for model prediction
                x_pred = np.array((np.linspace(6,16, 50), np.linspace(6,16, 50)))
                x_pred[0,x_pred[0,:]>midFreq] = 0
                x_pred[1,x_pred[1,:]<midFreq] = 0
                x_pred = x_pred.transpose()
                x_pred = sm.add_constant(x_pred)
                y_pred = model.predict(x_pred)

                #xNoGo_fit = np.linspace(6,16, 50)
                #yNoGo_fit = self.softmax(result_NoGo.x, xNoGo_fit-midFreq)

                psyCurve = StartPlots()
                psyCurve.ax.scatter(sortedFreqTotal[stiSortedInd], sortednumGo[stiSortedInd] / sortednumSound[stiSortedInd])
                psyCurve.ax.scatter(sortedFreqTotal[probeSortedInd], sortednumGo[probeSortedInd] / sortednumSound[probeSortedInd])
                psyCurve.ax.plot(np.linspace(6,16, 50), y_pred)
                #psyCurve.ax.plot(xNoGo_fit, yNoGo_fit)

                psyCurve.ax.plot([midFreq, midFreq], [0, 1], linestyle='--')

                # ax.legend()
                psyCurve.ax.set_xlabel('Sound (kHz)')
                psyCurve.ax.set_ylabel('Go rate')
                plt.show()

                psyCurve.save_plot('psychometric.svg', 'svg', save_path)
                psyCurve.save_plot('psychometric.tif', 'tif', save_path)
            else:
                self.saveData['L-fit'] = np.full((3), np.nan)

    def lick_rate(self, save_path,ifrun):

        plotname = os.path.join(save_path, 'lick rate.svg')
        if not os.path.exists(plotname) or ifrun:
            lickTimesH = np.array([])  # lick rate for Hit trials
            lickTimesFA =np.array([])   # lick rage for False alarm trials
            lickTimesProbe = np.array([])
            #lickSoundH = np.array(self.DF.sound_num[self.DF.reward==2])
            #lickSoundFA = np.array(self.DF.sound_num[self.DF.reward==-1])
            lickSoundH = np.array([])
            lickSoundFA = np.array([])
            lickSoundProbe = np.array([])

            binSize = 0.05  # use a 0.05s window for lick rate
            edges = np.arange(0 + binSize / 2, 5 - binSize / 2, binSize)

            for tt in range(self.trialN):
                if self.DF.trialType[tt] == 2:
                    lickTimesH = np.concatenate((lickTimesH, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                    lickSoundH = np.concatenate((lickSoundH, np.ones(len(np.array(self.DF.licks[tt])))*self.DF.sound_num[tt]))
                elif self.DF.trialType[tt] == -1:
                    lickTimesFA = np.concatenate((lickTimesFA, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                    lickSoundFA = np.concatenate(
                        (lickSoundFA, np.ones(len(np.array(self.DF.licks[tt]))) * self.DF.sound_num[tt]))
                elif self.DF.trialType[tt] == -3:
                    lickTimesProbe = np.concatenate((lickTimesProbe, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                    lickSoundProbe = np.concatenate(
                        (lickSoundProbe, np.ones(len(np.array(self.DF.licks[tt]))) * self.DF.sound_num[tt]))

            lickRateH = np.zeros((len(edges), 4))
            lickRateFA = np.zeros((len(edges), 4))
            lickRateProbe = np.zeros((len(edges), 8))

            for ee in range(len(edges)):
                for ssH in range(4):
                    lickRateH[ee,ssH] = sum(
                        np.logical_and(lickTimesH[lickSoundH==(ssH+1)] <= edges[ee] + binSize / 2, lickTimesH[lickSoundH==(ssH+1)] > edges[ee] - binSize / 2)) / (
                                        binSize * sum(np.logical_and(np.array(self.DF.trialType == 2), np.array(self.DF.sound_num)==(ssH+1))))
                for ssFA in range(4):
                    lickRateFA[ee, ssFA] = sum(
                        np.logical_and(lickTimesFA[lickSoundFA == (ssFA + 5)] <= edges[ee] + binSize / 2,
                                       lickTimesFA[lickSoundFA == (ssFA + 5)] > edges[ee] - binSize / 2)) / (
                                                 binSize * sum(np.logical_and(np.array(self.DF.trialType == -1),
                                                                              np.array(self.DF.sound_num) == (ssFA + 5))))
                for ssProbe in range(8):
                    lickRateProbe[ee, ssProbe] = sum(np.logical_and(lickTimesProbe[lickSoundProbe == (ssProbe + 9)] <= edges[ee] + binSize / 2,
                                       lickTimesProbe[lickSoundProbe == (ssProbe + 9)] > edges[ee] - binSize / 2)) / (
                                                   binSize * sum(np.logical_and(np.array(self.DF.trialType == -3),
                                                                                np.array(self.DF.sound_num) == (ssProbe + 9))))

            # save data
            self.saveData['lickRate'] = pd.DataFrame({'edges': edges})
            for ss in np.arange(1,17):
                if ss < 5: # hit
                    self.saveData['lickRate'][str(ss)] = lickRateH[:,ss-1]
                elif ss >= 5 and ss < 9:
                    self.saveData['lickRate'][str(ss)] = lickRateFA[:,ss-5]
                else:
                    self.saveData['lickRate'][str(ss)] = lickRateProbe[:,ss-9]

            # plot the response time distribution in hit/false alarm trials
            lickRate = StartPlots()

            lickRate.ax.plot(edges, np.nansum(lickRateH, axis=1))
            lickRate.ax.plot(edges, np.nansum(lickRateFA, axis=1))
            lickRate.ax.plot(edges, np.nansum(lickRateProbe, axis=1))

            lickRate.ax.set_xlabel('Time from cue (s)')
            lickRate.ax.set_ylabel('Frequency (Hz)')
            lickRate.ax.set_title('Lick rate (Hz)')

            lickRate.ax.legend(['Hit', 'False alarm','Probe lick'])

            plt.show()

            lickRate.save_plot('lick rate.svg', 'svg', save_path)
            lickRate.save_plot('lick rate.tif', 'tif', save_path)
            # separate the lick rate into different frequencies
            # fig, axs = plt.subplots(2, 4, figsize=(8, 8), sharey=True)
            #
            # # plot hit
            # for ii in range(4):
            #     axs[0, ii].plot(edges, lickRateH[:,ii])
            #     axs[0, ii].set_title(['Sound # ', str(ii + 1)])
            #
            # # plot false alarm
            # for jj in range(4):
            #     axs[1, jj].plot(edges, lickRateFA[:, jj])
            #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
            #
            # plt.subplots_adjust(top=0.85)
            # plt.show()

    def response_time(self, save_path,ifrun):
        """
        aligned_to: time point to be aligned. cue/outcome/licks
        """

        plotname = os.path.join(save_path, 'Response time.svg')
        if not os.path.exists(plotname) or ifrun:

            rt = np.zeros(self.trialN)

            for tt in range(self.trialN):
                rt[tt] = self.DF.first_lick_in[tt] - self.DF.onset[tt]

            # plot the response time distribution in hit/false alarm trials
            rtPlot = StartPlots()
            rtHit, bins, _ = rtPlot.ax.hist(rt[np.array(self.DF.trialType) == 2], bins=100, range=[0, 0.5], density=True)
            rtFA, _, _ = rtPlot.ax.hist(rt[np.array(self.DF.trialType) == -1], bins=bins, density=True)
            #_ = rtPlot.ax.hist(rt[np.array(self.DF.choice) == -3], bins=bins, density=True)

            # save the data
            self.saveData['rt'] = pd.DataFrame({'rtHit':rtHit, 'rtFA': rtFA, 'bin': bins[1:]})
            self.saveData['rtbyTrial'] = rt

            rtPlot.ax.set_xlabel('Response time (s)')
            rtPlot.ax.set_ylabel('Frequency (%)')
            rtPlot.ax.set_title('Response time (s)')

            rtPlot.ax.legend(['Hit', 'False alarm'])

            plt.show()

            rtPlot.save_plot('Response time.svg', 'svg', save_path)
            rtPlot.save_plot('Response time.tif', 'tif', save_path)
            # separate the response time into different frequencies
            # fig, axs = plt.subplots(2,4,figsize=(8, 8), sharey=True)
            #
            # # plot hit
            # for ii in range(4):
            #     if ii == 0:
            #         _, bins, _ = axs[0,ii].hist(rt[np.logical_and(np.array(self.DF.reward) == 2, self.DF.sound_num.array==(ii+1))], bins=50, range=[0, .5])
            #     else:
            #         _ = axs[0,ii].hist(rt[np.logical_and(np.array(self.DF.reward) == 2, self.DF.sound_num.array==(ii+1))], bins=bins)
            #     axs[0, ii].set_title(['Sound # ', str(ii+1)])
            #
            # # plot false alarm
            # for jj in range(4):
            #     _ = axs[1, jj].hist(rt[np.logical_and(np.array(self.DF.reward) == -1, self.DF.sound_num.array == (jj + 5))],bins=bins)
            #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
            #
            # plt.subplots_adjust(top=0.85)
            # plt.show()

    def ITI_distribution(self, save_path, ifrun):

        plotname = os.path.join(save_path, 'ITI.svg')
        if not os.path.exists(plotname) or ifrun:

            ITIH = []  # lick rate for Hit trials
            ITIFA = []  # lick rage for False alarm trials
            ITIProbe = []

            ITISoundH = np.array(self.DF.sound_num[np.logical_and(self.DF.trialType==2, self.DF.trial!=self.trialN)])
            ITISoundFA = np.array(self.DF.sound_num[np.logical_and(self.DF.trialType==-1,self.DF.trial!=self.trialN)])

            binSize = 0.05  # use a 0.05s window for lick rate
            edges = np.arange(0 + binSize / 2, 20- binSize / 2, binSize)

            for tt in range(self.trialN-1):
                if self.DF.trialType[tt] == 2:

                    ITIH.append(self.DF.onset[tt+1] - self.DF.outcome[tt])


                elif self.DF.trialType[tt] == -1:
                    ITIFA.append(self.DF.onset[tt+1] - self.DF.outcome[tt])


            # convert to np.arrays
            ITIH = np.array(ITIH)
            ITIFA = np.array(ITIFA)

            ITIRateH = np.zeros((len(edges), 4))
            ITIRateFA = np.zeros((len(edges), 4))

            for ee in range(len(edges)):
                for ssH in range(4):
                    ITIRateH[ee, ssH] = sum(
                        np.logical_and(ITIH[ITISoundH == (ssH + 1)] <= edges[ee] + binSize / 2,
                                       ITIH[ITISoundH == (ssH + 1)] > edges[ee] - binSize / 2))
                for ssFA in range(4):
                    ITIRateFA[ee, ssFA] = sum(
                        np.logical_and(ITIFA[ITISoundFA == (ssFA + 5)] <= edges[ee] + binSize / 2,
                                       ITIFA[ITISoundFA == (ssFA + 5)] > edges[ee] - binSize / 2))

            # plot
            ITIPlot = StartPlots()

            ITIPlot.ax.plot(edges, np.sum(ITIRateH, axis=1))
            ITIPlot.ax.plot(edges, np.sum(ITIRateFA, axis=1))

            ITIPlot.ax.set_xlabel('ITI duration (s)')
            ITIPlot.ax.set_ylabel('Trials')
            ITIPlot.ax.set_title('ITI distribution')

            ITIPlot.ax.legend(['Hit', 'False alarm'])

            plt.show()

            ITIPlot.save_plot('ITI.svg', 'svg', save_path)
            ITIPlot.save_plot('ITI.tif', 'tif', save_path)
            # # separate the lick rate into different frequencies
            # fig, axs = plt.subplots(2, 4, figsize=(8, 8), sharey=True)
            #
            # # plot hit
            # for ii in range(4):
            #     axs[0, ii].plot(edges, ITIRateH[:, ii])
            #     axs[0, ii].set_title(['Sound # ', str(ii + 1)])
            #
            # # plot false alarm
            # for jj in range(4):
            #     axs[1, jj].plot(edges, ITIRateFA[:, jj])
            #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
            #
            # plt.subplots_adjust(top=0.85)
            # plt.show()


    def running_aligned(self, aligned_to, save_path, ifrun):
        """
        aligned_to: reference time point. onset/outcome/lick
        """
        # aligned to cue onset and interpolate the results
        if 'running_time' in self.DF.columns:
            plotname = os.path.join(save_path, 'Running' + aligned_to + '.svg')
            if not os.path.exists(plotname) or ifrun:

                interpT = np.arange(-3,5,0.1)
                numBoot = 1000

                if aligned_to == "onset": # aligned to cue time
                    run_aligned = np.full((len(interpT), self.trialN), np.nan)

                    for tt in range(self.trialN-1):
                        speed = np.array(self.DF.running_speed[tt])
                        speedT = np.array(self.DF.running_time[tt])
                        if speed.size != 0:
                            t = speedT - self.DF.onset[tt]
                            y = speed
                            y_interp = np.interp(interpT, t, y)
                            run_aligned[:,tt] = y_interp

                        # bootstrap
                    BootH = bootstrap(run_aligned[:, self.DF.trialType == 2], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootFA = bootstrap(run_aligned[:, self.DF.trialType == -1], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootMiss = bootstrap(run_aligned[:, self.DF.trialType == -2], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootCorRej = bootstrap(run_aligned[:, self.DF.trialType == 0], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootProbeLick = bootstrap(run_aligned[:, self.DF.trialType == -3], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootProbeNoLick = bootstrap(run_aligned[:, self.DF.trialType == -4], dim=1, dim0 = len(interpT),n_sample=numBoot)


                elif aligned_to == 'outcome':
                    run_aligned = np.full((len(interpT), self.trialN), np.nan)
                    for tt in range(self.trialN-1):
                        speed = np.array(self.DF.running_speed[tt])
                        speedT = np.array(self.DF.running_time[tt])
                        if speed.size != 0:
                            t = speedT - self.DF.outcome[tt]
                            y = speed
                            y_interp = np.interp(interpT, t, y)
                            run_aligned[:,tt] = y_interp
                        # bootstrap
                    BootH = bootstrap(run_aligned[:, self.DF.trialType == 2], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootFA = bootstrap(run_aligned[:, self.DF.trialType == -1], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootMiss = bootstrap(run_aligned[:, self.DF.trialType == -2], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootCorRej = bootstrap(run_aligned[:, self.DF.trialType == 0], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootProbeLick = bootstrap(run_aligned[:, self.DF.trialType == -3], dim=1, dim0 = len(interpT),n_sample=numBoot)
                    BootProbeNoLick = bootstrap(run_aligned[:, self.DF.trialType == -4], dim=1, dim0 = len(interpT),n_sample=numBoot)


                elif aligned_to == 'licks':
                    run_aligned = []
                    for tt in range(self.trialN-1):
                            # loop through licks
                        numLicks = len(self.DF.licks[tt])
                        temp_aligned = np.full((len(interpT), numLicks), np.nan)
                        for ll in range(numLicks):
                            speed = np.array(self.DF.running_speed[tt])
                            speedT = np.array(self.DF.running_time[tt])
                            if speed.size != 0:
                                t = speedT - self.DF.licks[tt][ll]
                                y = speed
                                y_interp = np.interp(interpT, t, y)
                                temp_aligned[:,ll] = y_interp

                        run_aligned.append(temp_aligned)

                        # bootstrap
                    BootH = bootstrap(self.concat_data(run_aligned, 2), dim=1, dim0 = len(interpT), n_sample=numBoot)
                    BootFA = bootstrap(self.concat_data(run_aligned, -1), dim=1, dim0 = len(interpT), n_sample=numBoot)
                    BootMiss = bootstrap(self.concat_data(run_aligned, -2), dim=1, dim0 = len(interpT), n_sample=numBoot)
                    BootCorRej = bootstrap(self.concat_data(run_aligned, 0), dim=1, dim0 = len(interpT), n_sample=numBoot)
                    BootProbeLick = bootstrap(self.concat_data(run_aligned, -3), dim=1, dim0 = len(interpT), n_sample=numBoot)
                    BootProbeNoLick = bootstrap(self.concat_data(run_aligned, -4), dim=1, dim0 = len(interpT), n_sample=numBoot)

                # save the data
                self.saveData['run_' + aligned_to] = {'interpT': interpT, 'run_aligned': run_aligned}


                runPlot = StartSubplots(2,3,ifSharex=True, ifSharey=True)

                runPlot.fig.suptitle('Aligned to '+aligned_to)

                runPlot.ax[0,0].plot(interpT, BootH['bootAve'])
                runPlot.ax[0,0].fill_between(interpT, BootH['bootLow'], BootH['bootHigh'],alpha=0.2)
                runPlot.ax[0,0].set_title('Hit')
                runPlot.ax[0,0].set_ylabel('Running speed')

                runPlot.ax[0,1].plot(interpT, BootFA['bootAve'])
                runPlot.ax[0,1].fill_between(interpT, BootFA['bootLow'], BootFA['bootHigh'],alpha=0.2)
                runPlot.ax[0,1].set_title('False alarm')

                runPlot.ax[0,2].plot(interpT, BootMiss['bootAve'])
                runPlot.ax[0,2].fill_between(interpT, BootMiss['bootLow'], BootMiss['bootHigh'],alpha=0.2)
                runPlot.ax[0,2].set_title('Miss')

                runPlot.ax[1,0].plot(interpT, BootCorRej['bootAve'])
                runPlot.ax[1,0].fill_between(interpT, BootCorRej['bootLow'], BootCorRej['bootHigh'], alpha=0.2)
                runPlot.ax[1,0].set_title('Correct rejection')
                runPlot.ax[1,0].set_xlabel('Time (s)')
                runPlot.ax[1,0].set_ylabel('Running speed')

                runPlot.ax[1,1].plot(interpT, BootProbeLick['bootAve'])
                runPlot.ax[1,1].fill_between(interpT, BootProbeLick['bootLow'], BootProbeLick['bootHigh'], alpha=0.2)
                runPlot.ax[1,1].set_title('Probe lick')
                runPlot.ax[1,1].set_xlabel('Time (s)')

                runPlot.ax[1,2].plot(interpT, BootProbeNoLick['bootAve'])
                runPlot.ax[1,2].fill_between(interpT, BootProbeNoLick['bootLow'], BootProbeNoLick['bootHigh'], alpha=0.2)
                runPlot.ax[1,2].set_title('Probe nolick')
                runPlot.ax[1,2].set_xlabel('Time (s)')

                runPlot.save_plot('Running' + aligned_to + '.svg', 'svg', save_path)
                runPlot.save_plot('Running' + aligned_to + '.tif', 'tif', save_path)
    ### analysis methods for behavior

    def save_analysis(self, save_path, ifrun):
        # save the analysis result
        # save the analysis result
        # Open a file for writing
        save_file = os.path.join(save_path, 'behAnalysis.pickle')

        if not os.path.exists(save_file) or ifrun:

            with open(save_file, 'wb') as f:
                # Use pickle to dump the dictionary into the file
                pickle.dump(self.saveData, f)

    def fit_softmax(self, x, y):
        # Fit the softmax function to the data using scipy.optimize.minimize
        result = minimize(self.neg_log_likelihood, [0.5], args=(x, y))


    #define the softmax function
    def softmax(self, beta, x):

        return 1 / (1 + (np.exp(beta*x)))

    # Define the negative log-likelihood function
    def neg_log_likelihood(self, beta, x, y):
        p = self.softmax(beta, x)
        return -np.sum(y * np.log(p))

    def concat_data(self, data, outcome):
        # concatenate a list whose elements have different size
        # extract the trials with certain outcome and concatenate the running speed aligned to licks
        trialInd = [i for i, e in enumerate(self.DF.trialType) if e == outcome]
        output = np.array([])
        for tt in trialInd:
            if tt < self.trialN-1:
                output = data[tt] if not output.size else np.concatenate((output, data[tt]),axis=1)

        return output

class GoNogoBehaviorSum:
    # class to make summary analysis and plot

    def __init__(self,root_dir):
        # start with the root directory, generate beh_df
        raw_beh = 'processed_behavior'
        raw_fluo = 'raw_imaging'

        # specify saved files
        analysis_dir = 'analysis'
        analysis_beh = 'behavior'

        animals = os.listdir(os.path.join(root_dir, raw_beh))

        # initialize the dataframe
        columns = ['file', 'file_path', 'date', 'subject', 'age', 'saved_dir']
        beh_df = pd.DataFrame(columns=columns)

        # go through the files to update the dataframe
        for animal in animals:
            animal_path = os.path.join(root_dir, raw_beh, animal)
            sessions = glob.glob(os.path.join(animal_path, animal + '*' + '-behaviorLOG.mat'))
            Ind = 0
            for session in sessions:
                separated = os.path.basename(session).split("-")
                data = pd.DataFrame({
                    'file': os.path.basename(session),
                    'file_path': session,
                    'date': separated[1],
                    'subject': animal,
                    'age': animal[0:3],
                    'saved_dir': os.path.join(root_dir, analysis_dir, analysis_beh, animal, separated[1])
                }, index=[Ind])
                Ind = Ind + 1
                beh_df = pd.concat([beh_df, data])

        self.beh_df = beh_df
        self.beh_dict = dict()

    def process_singleSession(self, ifrun=False):
        # go through individual sessions and run analysis, save data and plot
        # ifrun: if True, run the analysis regardless a file already exist
        nFiles = len(self.beh_df['file'])
        for f in tqdm(range(nFiles)):
            animal = self.beh_df.iloc[f]['subject']
            session = self.beh_df.iloc[f]['date']
            input_path = self.beh_df.iloc[f]['file_path']
            x = GoNogoBehaviorMat(animal, session, input_path)
            x.to_df()
            output_path = self.beh_df.iloc[f]['saved_dir']
            plot_path = os.path.join(output_path, 'beh_plot')

            # x.beh_cut(plot_path)
            # run analysis_beh
            x.d_prime()

            # make plot
            x.beh_session(plot_path, ifrun)
            x.psycho_curve(plot_path, ifrun)
            x.lick_rate(plot_path, ifrun)
            x.ITI_distribution(plot_path, ifrun)
            x.response_time(plot_path, ifrun)
            x.running_aligned('onset', plot_path, ifrun)
            x.running_aligned('outcome', plot_path, ifrun)
            x.running_aligned('licks', plot_path, ifrun)

            plt.close('all')
            x.save_analysis(output_path, ifrun)

    def read_data(self):
        # read all analyzed data of individual sessions and save to a dictionary

        nFiles = self.beh_df.shape[0]
        self.trainSti = np.zeros(nFiles) # number of stimulus used
        self.animalList = np.unique(self.beh_df['subject'])
        # should be 2 (1 go, 1 no go)

        ### initialize the concateChoice to store the choice and stimulus for every animal
        self.concateChoice = dict()
        for aa in self.animalList:
            self.concateChoice[aa] = dict()
            self.concateChoice[aa]['sound'] = []
            self.concateChoice[aa]['session'] = []
            self.concateChoice[aa]['trialType'] = []
            self.concateChoice[aa]['numSound'] = []
            self.concateChoice[aa]['respT'] = []
            # concatenate files
            animalInd =  [i for i, val in enumerate(self.beh_df['subject']) if val == aa]
            tempInd = 0
            numSti = 0  # the variable is used to determine when the new stimulus was first introduced
            # if the animal experienced 8 stimulus in the previous session, then the value is 8 for all following session ,
            # regardless on how many stimulus it experienced in following sessions

            for aInd in animalInd:
                save_file = os.path.join(self.beh_df.iloc[aInd]['saved_dir'], 'behAnalysis.pickle')
                # load pickle file
                with open(save_file, 'rb') as pf:
                    # Load the data from the pickle file
                    my_data = pickle.load(pf)
                    pf.close()

                self.concateChoice[aa]['sound'] = np.append(self.concateChoice[aa]['sound'],
                                                            np.array(my_data['behDF']['sound_num'].values))
                self.concateChoice[aa]['session'] = np.append(self.concateChoice[aa]['session'],
                                                              np.ones(len(my_data['behDF']['sound_num']))*tempInd)
                self.concateChoice[aa]['trialType'] = np.append(self.concateChoice[aa]['trialType'] ,
                                                             np.array(my_data['behDF']['trialType'].values))
                self.concateChoice[aa]['respT'] = np.append(self.concateChoice[aa]['respT'] ,
                                                              my_data['rtbyTrial'])
                tempInd = tempInd + 1

            # once we have the complete sequence of stimulus presented, determine when a new stiumulus is first introduced

            # Use a set to get unique elements in any order
            uniqueSound = [] # this list contains all sound stimulus, ordered by the time of their first appearance
            self.concateChoice[aa]['numSound'] = []
            numSound = 0
            for val in self.concateChoice[aa]['sound']:
                if val not in uniqueSound:
                    # update number of sound experienced
                    if numSound < 2:
                        numSound = numSound + 1
                    elif numSound < 8 and val <= 4: # sounds are added in pairs
                        numSound = numSound + 2
                    #elif val > 8 and np.all(np.array(uniqueSound) <= 8):  # + 8 once, since probe stimulus are added at once
                    #    numSound = numSound + 8
                    uniqueSound.append(val)
                self.concateChoice[aa]['numSound'] = np.append(self.concateChoice[aa]['numSound'], numSound)

        # initialize variables
        for ff in tqdm(range(nFiles)):
            save_file = os.path.join(self.beh_df.iloc[ff]['saved_dir'],'behAnalysis.pickle')
            # load pickle file
            with open(save_file, 'rb') as pf:
                # Load the data from the pickle file
                my_data = pickle.load(pf)
                pf.close()
           # basic information: subject, session
            if ff == 0:
                self.beh_dict['animal'] = [None] * nFiles
                self.beh_dict['session'] = [None] * nFiles
                self.beh_dict['stiFreq'] = my_data['psycho-sti']

            self.beh_dict['animal'][ff] = my_data['behDF'].iloc[0]['animal']
            self.beh_dict['session'][ff] = my_data['behDF'].iloc[0]['session']
            self.trainSti[ff] = len(np.unique(my_data['behDF']['sound_num']))

           # initialize the summary dictionary if it is the first file
            for key in my_data.keys():
                if not key in ['behFull', 'behDF', 'rtbyTrial', 'psycho-sti'] and not 'run' in key:
                    # ignore the run_aligned for now, until coming up with how to summarize it

                    if isinstance(my_data[key], np.ndarray):
                        # L-fit and psycho_data are the only array for now
                        if ff == 0:
                            self.beh_dict[key] = np.zeros((len(my_data[key]), nFiles))
                        # set the value
                        self.beh_dict[key][:,ff] = my_data[key]

                    elif isinstance(my_data[key], pd.DataFrame):
                            # respone time/lick rate
                        if ff == 0:
                            dfShape = my_data[key].shape
                            self.beh_dict[key] = np.zeros((dfShape[0], dfShape[1], nFiles))
                        # set values
                        self.beh_dict[key][:,:,ff] = my_data[key]
                    elif isinstance(my_data[key], dict):
                            # run aligns are dictionary
                        if 'interpT' in my_data[key].keys():
                            if ff == 0:
                                self.beh_dict['interpT_run'] = my_data[key]['interpT']
                                arrShape = my_data[key]['run_aligned'].shape
                                self.beh_dict[key] = np.zeros((arrShape[0], arrShape[1], nFiles))
                            # set value
                            self.beh_dict[key][:,:,ff] = my_data[key]['run_aligned']
                    else:
                        if ff == 0:
                            # a single value
                            self.beh_dict[key] = np.zeros((nFiles))
                        # set value
                        self.beh_dict[key][ff] = my_data[key]

    def plot_dP(self, save_path):
        # plot d-prime progress by animals
        # concatenate sessions from one animla, calculate running/block d-prime (50 trials)
        # also separated by number of stimulus presented
        # separate by age
        # also separate by training protocol (number of stimulus used)

        # go through every animal, align the behavior with number of sounds introduced

        # start with two sound stimulus first
        #time.sleep(1)
        stiNumList = [1, 2, 4, 6, 8]
        tStep = 50 # calculate the d-prime in block of 50 trials
        runningdPrime = {}#dict()

        for ss in range(len(stiNumList) - 1):
            runningdPrime[str(stiNumList[ss+1])] = {}#dict()
            for aa in self.animalList:

                tempInd = np.logical_and(self.concateChoice[aa]['numSound'] > stiNumList[ss], self.concateChoice[aa]['numSound'] <= stiNumList[ss+1])
                tempChoice = self.concateChoice[aa]['trialType'][tempInd]

                runningdPrime[str(stiNumList[ss+1])][aa] =[]

                nBlocks = len(tempChoice)//tStep + 1
                if nBlocks == 1:  # if less than 50 trials
                    Hit_rate = np.sum(tempChoice == 2) / np.sum(np.logical_or(tempChoice == 2,tempChoice == -2))
                    FA_rate = np.sum(tempChoice == -1) / np.sum(np.logical_or(tempChoice == 0,tempChoice == -1))
                    runningdPrime[str(stiNumList[ss+1])][aa].append(norm.ppf(self.check_rate(Hit_rate))
                                                                    - norm.ppf(self.check_rate(FA_rate)))
                else:
                    for bb in range(nBlocks):
                        startT = bb*tStep
                        endT= (bb+1)*tStep
                        if endT < len(tempChoice):
                            Hit_rate = np.sum(tempChoice[startT:endT] == 2) / np.sum(
                                np.logical_or(tempChoice[startT:endT] == 2,tempChoice[startT:endT]== -2))
                            FA_rate = np.sum(tempChoice[startT:endT] == -1) / np.sum(
                                np.logical_or(tempChoice[startT:endT] == -1,tempChoice[startT:endT]==0))
                        else:
                            Hit_rate = np.sum(tempChoice[startT:] == 2) / np.sum(
                                np.logical_or(tempChoice[startT:] == 2, tempChoice[startT:] == -2))
                            FA_rate = np.sum(tempChoice[startT:] == -1) / np.sum(
                                np.logical_or(tempChoice[startT:] == -1, tempChoice[startT:] == 0))
                        runningdPrime[str(stiNumList[ss+1])][aa] = np.append(
                            runningdPrime[str(stiNumList[ss+1])][aa],norm.ppf(self.check_rate(Hit_rate)) - norm.ppf(self.check_rate(FA_rate)))

        # reorganize the data into dataframe
        # iterate through every stage (number of stimulus)
        dPrimeMat_start = {} # aligned to the start of adding sound stimulus
        dPrimeMat_end = {}   # aligned to the end of adding sound stimulus


        for key in runningdPrime.keys():
            max_length = max(len(v) for v in runningdPrime[key].values())
            arr_1 = np.empty((max_length, len(runningdPrime[key])), dtype=float)
            arr_1[:] = np.nan

            arr_2 = np.empty((max_length, len(runningdPrime[key])), dtype=float)
            arr_2[:] = np.nan

            for i, kk in enumerate(sorted(runningdPrime[key].keys())):
                arr_1[:len(runningdPrime[key][kk]), i] = runningdPrime[key][kk]
                arr_2[-len(runningdPrime[key][kk]):, i] = runningdPrime[key][kk]
            dPrimeMat_start[key] = arr_1
            dPrimeMat_end[key] = arr_2

        # separate animals into adult and juvenile

        adtInd = [i for i in range(len(self.animalList)) if 'ADT' in self.animalList[i]]
        juvInd = [i for i in range(len(self.animalList)) if 'JUV' in self.animalList[i]]
        dP_plot = StartPlots()

        startX = 0
        adtColor = (255/255, 189/255, 53/255)
        juvColor = (63/255, 167/255, 150/255)
        x_Ticks = []  # xticks corresponding to the original figure
        x_Ticks_show = []  # xticks showing the number of blocks from the aligne point
        for key in dPrimeMat_start.keys():

            # plot vertical line showing the stimulus transition
            dP_plot.ax.axvline(x=startX-2.5, linestyle='--', color='black', linewidth=0.5)
            plt.text(startX-2, 4, key)

            # plot the part aligned to the start
            adtD_1 = dPrimeMat_start[key][:,adtInd]
            juvD_1 = dPrimeMat_start[key][:,juvInd]

            # only look at data with more than two animals
            adtI_1 = np.where(np.count_nonzero(~np.isnan(adtD_1), axis=1) > 2)[0]
            juvI_1 = np.where(np.count_nonzero(~np.isnan(juvD_1), axis=1) > 2)[0]

            plotLength = min(20, len(adtI_1),len(juvI_1))
            xAxis = np.arange(startX, startX + plotLength)

            adtPlot = adtD_1[adtI_1[:plotLength],:]
            juvPlot = juvD_1[juvI_1[:plotLength],:]

            # two-tail paird t-test for equal means
            pVal = np.zeros(plotLength)
            for i in range(plotLength):
                _, pVal[i] = ttest_ind(adtPlot[i,:], juvPlot[i,:],nan_policy='omit')

            dP_plot.ax.plot(xAxis, np.nanmean(adtPlot,1), color=adtColor, label='Adult')
            se_11 = np.nanstd(adtPlot,1)/np.sqrt(np.count_nonzero(~np.isnan(adtPlot), axis=1))
            dP_plot.ax.fill_between(xAxis, np.nanmean(adtPlot,1)-se_11,
                                   np.nanmean(adtPlot,1)+se_11, color=adtColor,alpha=0.2, label='_nolegend_')

            dP_plot.ax.plot(xAxis, np.nanmean(juvPlot,1), color=juvColor, label='Juvenile')
            se_12 = np.nanstd(juvPlot, 1) / np.sqrt(np.count_nonzero(~np.isnan(juvPlot), axis=1))
            dP_plot.ax.fill_between(xAxis, np.nanmean(juvPlot,1)-se_12,
                                   np.nanmean(juvPlot,1)+se_12, color=juvColor,alpha=0.2, label='_nolegend_')

            # plot p
            for tt in range(plotLength):
                if pVal[tt] < 0.05:
                    dP_plot.ax.plot(xAxis[tt] + 1 * np.array([-0.5, 0.5]), [4, 4],
                                                   color=(1, 0, 0), linewidth=5)

            # set xticks
            x_Ticks = np.append(x_Ticks, xAxis[0::3])
            x_Ticks_show = np.append(x_Ticks_show, (xAxis[0::3]-startX).astype(str))

            startX = startX + plotLength + 5

            # plot the part aligned to the end
            adtD_2 = dPrimeMat_end[key][:,adtInd]
            juvD_2 = dPrimeMat_end[key][:,juvInd]

            # only look at data with more than two animals
            adtI_2 = np.where(np.count_nonzero(~np.isnan(adtD_2), axis=1) > 2)[0]
            juvI_2 = np.where(np.count_nonzero(~np.isnan(juvD_2), axis=1) > 2)[0]
            # t-test for d-prime data
            #adtBoot = bootstrap(adtD, 1, adtD.shape[0], 1000)
            #juvBoot = bootstrap(juvD, 1, juvD.shape[0], 1000)
            plotLength = min(20, len(adtI_2),len(juvI_2))
            xAxis = np.arange(startX, startX + plotLength)

            adtPlot = adtD_2[adtI_2[-plotLength:],:]
            juvPlot = juvD_2[juvI_2[-plotLength:],:]

            # two-tail t-test for equal means
            pVal = np.zeros(plotLength)
            for i in range(plotLength):
                _, pVal[i] = ttest_ind(adtPlot[i,:], juvPlot[i,:],nan_policy='omit')

            dP_plot.ax.plot(xAxis, np.nanmean(adtPlot,1), color=adtColor, label='Adult')
            se_21 = np.nanstd(adtPlot,1)/np.sqrt(np.count_nonzero(~np.isnan(adtPlot), axis=1))
            dP_plot.ax.fill_between(xAxis, np.nanmean(adtPlot,1)-se_21,
                                   np.nanmean(adtPlot,1)+se_21, color=adtColor,alpha=0.2, label='_nolegend_')

            dP_plot.ax.plot(xAxis, np.nanmean(juvPlot,1), color=juvColor, label='Juvenile')
            se_22 = np.nanstd(juvPlot,1)/np.sqrt(np.count_nonzero(~np.isnan(juvPlot), axis=1))
            dP_plot.ax.fill_between(xAxis, np.nanmean(juvPlot,1)-se_22,
                                   np.nanmean(juvPlot,1)+se_22, color=juvColor,alpha=0.2, label='_nolegend_')

            # plot significance
            for tt in range(plotLength):
                if pVal[tt] < 0.05:
                    dP_plot.ax.plot(xAxis[tt] + 1 * np.array([-0.5, 0.5]), [4, 4],
                                                   color=(1, 0, 0), linewidth=5)

            x_Ticks = np.append(x_Ticks, xAxis[0::3])
            x_Ticks_show = np.append(x_Ticks_show, (-xAxis[::-1][0::3]+startX).astype(str))

            startX = startX + plotLength + 5

        dP_plot.ax.set_xticks(x_Ticks)
        dP_plot.ax.set_xticklabels(x_Ticks_show,fontsize=8)
        dP_plot.ax.set_xlabel('Number of 50 trial blocks')
        dP_plot.ax.set_ylabel('Average d-prime')
        dP_plot.legend(['Stimulus','Adult', 'Juvenile'])

        plt.show()

        dP_plot.save_plot('Average d-prime.svg', 'svg', save_path)
        dP_plot.save_plot('Average d-prime.tif', 'tif', save_path)

    def plot_rt(self, cues, save_path):
        # plot the response time in a similar manner as the d-prime
        # separate into ADT/JUV, Hit/FA, training state, combined cues/separated cues
        # input: cues that need to be plotted
        #       e.g. all cues [1,2,3,4,5,6,7,8]
        #            single cue pairs [2,7]
        #            probe cues (9.17,1)
        stiNumList = [1, 2, 4, 6, 8]
        tStep = 50 # calculate the average response time in block of 50 trials
        runningRTHit_ADT_start = dict()
        runningRTFA_ADT_start = dict()
        runningRTHit_JUV_start = dict()
        runningRTFA_JUV_start = dict()

        runningRTHit_ADT_end= dict()
        runningRTFA_ADT_end = dict()
        runningRTHit_JUV_end = dict()
        runningRTFA_JUV_end = dict()

        for ss in range(len(stiNumList) - 1):
            runningRTHit_ADT_start[str(stiNumList[ss+1])] = dict()
            runningRTFA_ADT_start[str(stiNumList[ss+1])] = dict()

            runningRTHit_JUV_start[str(stiNumList[ss+1])] = dict()
            runningRTFA_JUV_start[str(stiNumList[ss+1])] = dict()

            runningRTHit_ADT_end[str(stiNumList[ss+1])] = dict()
            runningRTFA_ADT_end[str(stiNumList[ss+1])] = dict()

            runningRTHit_JUV_end[str(stiNumList[ss+1])] = dict()
            runningRTFA_JUV_end[str(stiNumList[ss+1])] = dict()

            # look at 20 blocks maximum
            for bb in tqdm(range(20)):
                for iaa, aa in enumerate(self.animalList):

                    totalInd = np.arange(len(self.concateChoice[aa]['numSound'])) # all trials available for animal aa
                    blockInd = totalInd[np.logical_and(self.concateChoice[aa]['numSound'] > stiNumList[ss],
                                                   self.concateChoice[aa]['numSound'] <= stiNumList[ss+1])] # trials available for given block ss

                    # all available trials for cues/hit/FA of animal aa
                    cueInd = [idx for idx in range(len(self.concateChoice[aa]['numSound'])) if self.concateChoice[aa]['sound'][idx] in cues]
                    hitInd = [idx for idx in range(len(self.concateChoice[aa]['trialType'])) if self.concateChoice[aa]['trialType'][idx] > 0]
                    FAInd = [idx for idx in range(len(self.concateChoice[aa]['trialType'])) if self.concateChoice[aa]['trialType'][idx] == -1]

                    hitFAIndSet = set(hitInd) | set(FAInd) # combine FA and hit trials to get 50 trial blocks

                    # trials needed: within block and cue, hit + false alarm trials
                    tempIndSet = hitFAIndSet & set(blockInd) & set(cueInd)
                    tempRT = self.concateChoice[aa]['respT'][list(tempIndSet)]
                    tempInd = totalInd[list(tempIndSet)]

        # for probe trials, separate probe trials into go probe trials and no go probe trials
        # since there are only 1% probe trials, consider all probe trials together
                    goProbe = [9,10,11,12]
                    nogoProbe = [13,14,15,16]

                    if 9 in cues: # if deal with probe trials
                        hitInd = [idx for idx in range(len(self.concateChoice[aa]['trialType'])) if
                                  (self.concateChoice[aa]['sound'][idx] in goProbe and self.concateChoice[aa]['trialType'][idx] == -3)]   # hit here refers to go probe trials
                        FAInd = [idx for idx in range(len(self.concateChoice[aa]['trialType'])) if
                                 (self.concateChoice[aa]['sound'][idx] in nogoProbe and self.concateChoice[aa]['trialType'][idx] == -3)]  # false alarm here refers to nogo probe trials

                        hitFAIndSet = set(hitInd) | set(FAInd)  # combine FA and hit trials to get 50 trial blocks

                        # trials needed: within block and cue, go/no go probe trials
                        tempIndSet = hitFAIndSet & set(blockInd) & set(cueInd)
                        tempRT = self.concateChoice[aa]['respT'][list(tempIndSet)]
                        tempInd = totalInd[list(tempIndSet)]

                 # get the trials from the correct block, with the right cues, aligned to the block start
                    if iaa == 0:
                        runningRTHit_ADT_start[str(stiNumList[ss + 1])][bb] = np.array([])
                        runningRTFA_ADT_start[str(stiNumList[ss + 1])][bb] = np.array([])

                        runningRTHit_JUV_start[str(stiNumList[ss + 1])][bb] = np.array([])
                        runningRTFA_JUV_start[str(stiNumList[ss + 1])][bb] = np.array([])

                        runningRTHit_ADT_end[str(stiNumList[ss + 1])][bb] = np.array([])
                        runningRTFA_ADT_end[str(stiNumList[ss + 1])][bb] = np.array([])

                        runningRTHit_JUV_end[str(stiNumList[ss + 1])][bb] = np.array([])
                        runningRTFA_JUV_end[str(stiNumList[ss + 1])][bb] = np.array([])

                    # index to align to the block start
                    startT_start = bb*tStep
                    endT_start= (bb+1)*tStep

                    # index to align to the block end
                    startT_end = len(tempRT) - (bb+1)*tStep
                    endT_end= len(tempRT) - bb*tStep-1

                    # align to start
                    if endT_start < len(tempRT):
                        hitI_start = [list(tempInd).index(elem) for elem in
                                      list(set(tempInd[startT_start:endT_start]) & set(hitInd))]
                        FAI_start = [list(tempInd).index(elem) for elem in
                                     list(set(tempInd[startT_start:endT_start]) & set(FAInd))]
                    else:
                        hitI_start = [list(tempInd).index(elem) for elem in
                                      list(set(tempInd[startT_start:]) & set(hitInd))]
                        FAI_start = [list(tempInd).index(elem) for elem in
                                     list(set(tempInd[startT_start:]) & set(FAInd))]

                    # align to end
                    if startT_end > -1:
                        hitI_end = [list(tempInd).index(elem) for elem in
                                list(set(tempInd[startT_end:endT_end]) & set(hitInd))]
                        FAI_end = [list(tempInd).index(elem) for elem in
                               list(set(tempInd[startT_end:endT_end]) & set(FAInd))]
                    elif startT_end <= -1 and endT_end > -1:
                        hitI_end = [list(tempInd).index(elem) for elem in
                                    list(set(tempInd[:endT_end]) & set(hitInd))]
                        FAI_end = [list(tempInd).index(elem) for elem in
                                   list(set(tempInd[:endT_end]) & set(FAInd))]
                    elif endT_end <= -1:
                        hitI_end = []
                        FAI_end = []

                    Hitresp_start = tempRT[hitI_start]
                    FAresp_start = tempRT[FAI_start]

                    Hitresp_end = tempRT[hitI_end]
                    FAresp_end = tempRT[FAI_end]
                # get the trials from the correct block, with the right cues, aligned to the block end


                    if 'ADT' in aa:
                        runningRTHit_ADT_start[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTHit_ADT_start[str(stiNumList[ss + 1])][bb],Hitresp_start)
                        runningRTFA_ADT_start[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTFA_ADT_start[str(stiNumList[ss + 1])][bb],FAresp_start)

                        runningRTHit_ADT_end[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTHit_ADT_end[str(stiNumList[ss + 1])][bb],Hitresp_end)
                        runningRTFA_ADT_end[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTFA_ADT_end[str(stiNumList[ss + 1])][bb],FAresp_end)

                    elif 'JUV' in aa:
                        runningRTHit_JUV_start[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTHit_JUV_start[str(stiNumList[ss + 1])][bb], Hitresp_start)
                        runningRTFA_JUV_start[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTFA_JUV_start[str(stiNumList[ss + 1])][bb], FAresp_start)

                        runningRTHit_JUV_end[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTHit_JUV_end[str(stiNumList[ss + 1])][bb],Hitresp_end)
                        runningRTFA_JUV_end[str(stiNumList[ss + 1])][bb] = np.append(
                            runningRTFA_JUV_end[str(stiNumList[ss + 1])][bb],FAresp_end)

        # reorganize the data into dataframe
        # iterate through every stage (number of stimulus)

        if not 9 in cues:  # for go no-go trials with cue 1-9
            # make a same plot for the final stage of [1
            dP_plot = StartPlots()

            startX = 0
            adtColor = (255/255, 189/255, 53/255)
            juvColor = (63/255, 167/255, 150/255)
            x_Ticks = []  # xticks corresponding to the original figure
            x_Ticks_show = []  # xticks showing the number of blocks from the aligne point
            for key in runningRTHit_ADT_start.keys():
                # caculate average and standard error for each response time
                ave_ADT_Hit_start, ste_ADT_Hit_start = self.get_meanste(runningRTHit_ADT_start[key])
                ave_ADT_FA_start, ste_ADT_FA_start = self.get_meanste(runningRTFA_ADT_start[key])

                ave_JUV_Hit_start, ste_JUV_Hit_start = self.get_meanste(runningRTHit_JUV_start[key])
                ave_JUV_FA_start, ste_JUV_FA_start = self.get_meanste(runningRTFA_JUV_start[key])

                ave_ADT_Hit_end, ste_ADT_Hit_end = self.get_meanste(runningRTHit_ADT_end[key])
                ave_ADT_FA_end, ste_ADT_FA_end = self.get_meanste(runningRTFA_ADT_end[key])

                ave_JUV_Hit_end, ste_JUV_Hit_end = self.get_meanste(runningRTHit_JUV_end[key])
                ave_JUV_FA_end, ste_JUV_FA_end = self.get_meanste(runningRTFA_JUV_end[key])

                # plot vertical line showing the stimulus transition
                dP_plot.ax.axvline(x=startX-2.5, linestyle='--', color='black', linewidth=0.5)
                plt.text(startX-2, 4, key)

                # calculate the average response time

                plotLength = min(np.count_nonzero(~np.isnan(ave_ADT_Hit_start)), np.count_nonzero(~np.isnan(ave_ADT_FA_start)),
                                 np.count_nonzero(~np.isnan(ave_JUV_Hit_start)),np.count_nonzero(~np.isnan(ave_JUV_Hit_start)))
                xAxis = np.arange(startX, startX + plotLength)

                # statistical test, anova?
                #pVal = np.zeros(plotLength)
                #for i in range(plotLength):
                #    _, pVal[i] = ttest_ind(adtPlot[i,:], juvPlot[i,:],nan_policy='omit')

            # plots aligned to the start
                dP_plot.ax.plot(xAxis, ave_ADT_Hit_start[:plotLength], color=adtColor, label='Adult hit')
                dP_plot.ax.fill_between(xAxis, ave_ADT_Hit_start[:plotLength]-ste_ADT_Hit_start[:plotLength],
                                       ave_ADT_Hit_start[:plotLength]+ste_ADT_Hit_start[:plotLength],
                                    color=adtColor,alpha=0.2, label='_nolegend_')

                dP_plot.ax.plot(xAxis, ave_ADT_FA_start[:plotLength], linestyle = '--', color=adtColor, label='Adult FA')
                dP_plot.ax.fill_between(xAxis, ave_ADT_FA_start[:plotLength]-ste_ADT_FA_start[:plotLength],
                                       ave_ADT_FA_start[:plotLength]+ste_ADT_FA_start[:plotLength],
                                       color=adtColor,alpha=0.2, label='_nolegend_')

                dP_plot.ax.plot(xAxis, ave_JUV_Hit_start[:plotLength], color=juvColor, label='Juvinile hit')
                dP_plot.ax.fill_between(xAxis, ave_JUV_Hit_start[:plotLength]-ste_JUV_Hit_start[:plotLength],
                                       ave_JUV_Hit_start[:plotLength]+ste_JUV_Hit_start[:plotLength],
                                        color=juvColor,alpha=0.2, label='_nolegend_')

                dP_plot.ax.plot(xAxis, ave_JUV_FA_start[:plotLength], linestyle = '--', color=juvColor, label='Juvenile FA')
                dP_plot.ax.fill_between(xAxis, ave_JUV_FA_start[:plotLength]-ste_JUV_FA_start[:plotLength],
                                       ave_JUV_FA_start[:plotLength]+ste_JUV_FA_start[:plotLength],
                                       color=juvColor,alpha=0.2, label='_nolegend_')

                # plot p
                # for tt in range(plotLength):
                #    if pVal[tt] < 0.05:
                #        dP_plot.ax.plot(xAxis[tt] + 1 * np.array([-0.5, 0.5]), [4, 4],
                #                                       color=(1, 0, 0), linewidth=5)

                # set xticks
                x_Ticks = np.append(x_Ticks, xAxis[0::3])
                x_Ticks_show = np.append(x_Ticks_show, (xAxis[0::3]-startX).astype(str))

                startX = startX + plotLength + 5

            # plot the part aligned to the end

                # only look at data with more than two animals

                # t-test for d-prime data
                plotLength = min(np.count_nonzero(~np.isnan(ave_ADT_Hit_end)), np.count_nonzero(~np.isnan(ave_ADT_FA_end)),
                                 np.count_nonzero(~np.isnan(ave_JUV_Hit_end)),np.count_nonzero(~np.isnan(ave_JUV_Hit_end)))
                xAxis = np.arange(startX, startX + plotLength)

                if plotLength>0:
                    dP_plot.ax.plot(xAxis, ave_ADT_Hit_end[plotLength-1::-1], color=adtColor, label='Adult hit')
                    dP_plot.ax.fill_between(xAxis, ave_ADT_Hit_end[plotLength-1::-1]-ste_ADT_Hit_end[plotLength-1::-1],
                                           ave_ADT_Hit_end[plotLength-1::-1]+ste_ADT_Hit_end[plotLength-1::-1],
                                           color=adtColor,alpha=0.2, label='_nolegend_')

                    dP_plot.ax.plot(xAxis, ave_ADT_FA_end[plotLength-1::-1], linestyle = '--', color=adtColor, label='Adult FA')
                    dP_plot.ax.fill_between(xAxis, ave_ADT_FA_end[plotLength-1::-1]-ste_ADT_FA_end[plotLength-1::-1],
                                           ave_ADT_FA_end[plotLength-1::-1]+ste_ADT_FA_end[plotLength-1::-1],
                                           color=adtColor,alpha=0.2, label='_nolegend_')

                    dP_plot.ax.plot(xAxis, ave_JUV_Hit_end[plotLength-1::-1], color=juvColor, label='Juvinile hit')
                    dP_plot.ax.fill_between(xAxis, ave_JUV_Hit_end[plotLength-1::-1]-ste_JUV_Hit_end[plotLength-1::-1],
                                           ave_JUV_Hit_end[plotLength-1::-1]+ste_JUV_Hit_end[plotLength-1::-1],
                                            color=juvColor,alpha=0.2, label='_nolegend_')

                    dP_plot.ax.plot(xAxis, ave_JUV_FA_end[plotLength-1::-1], linestyle = '--', color=juvColor, label='Juvenile FA')
                    dP_plot.ax.fill_between(xAxis, ave_JUV_FA_end[plotLength-1::-1]-ste_JUV_FA_end[plotLength-1::-1],
                                           ave_JUV_FA_end[plotLength-1::-1]+ste_JUV_FA_end[plotLength-1::-1],
                                           color=juvColor,alpha=0.2, label='_nolegend_')


                x_Ticks = np.append(x_Ticks, xAxis[0::3])
                x_Ticks_show = np.append(x_Ticks_show, (-xAxis[::-1][0::3]+startX).astype(str))

                startX = startX + plotLength + 5

                dP_plot.ax.set_xticks(x_Ticks)
                dP_plot.ax.set_xticklabels(x_Ticks_show, fontsize=8)
                dP_plot.ax.set_xlabel('Number of 50 trial blocks')
                dP_plot.ax.set_ylabel('Average response time (s)')
                titletext = '-'.join(str(i) for i in cues)
                dP_plot.ax.set_title('Average response time cue(' + titletext + ')')
                dP_plot.legend(['Stimulus', 'Adult hit', 'Adult FA', 'Juvenile hit', 'Juvenile FA'])
                dP_plot.fig.set_figwidth(30)

                ## make a same violin plot as in probe trials
                ## with the last 2000 trials (assume they are relative stable behavior)
            plt.show()

            dP_plot.save_plot('Average response time cue(' + titletext + ').svg', 'svg', save_path)
            dP_plot.save_plot('Average response time cue(' + titletext + ').tif', 'tif', save_path)


        # plot end stage analysis of average response time for cues

                # plot go-probe/no-go probe in bar plot
        key = '8'
                # combine the blocks

            # for probe trials: these variable correpond to: adult (near) go cue; adult nogo cue; juv go cue; juv nogo cue
            # for task (1-8) trials: adult hit; adult FA; juv hit; juv FA

        ADTProbeGoRT = np.array([])
        ADTProbeNogoRT = np.array([])
        JUVProbeGoRT = np.array([])
        JUVProbeNogoRT = np.array([])

        for kk in runningRTHit_ADT_end[key].keys():
            if len(runningRTHit_ADT_end[key][kk])>0:
                ADTProbeGoRT = np.append(ADTProbeGoRT, runningRTHit_ADT_end[key][kk])
            if len(runningRTFA_ADT_end[key][kk]) > 0:
                ADTProbeNogoRT = np.append(ADTProbeNogoRT, runningRTFA_ADT_end[key][kk])
            if len(runningRTHit_JUV_end[key][kk]) > 0:
                JUVProbeGoRT = np.append(JUVProbeGoRT, runningRTHit_JUV_end[key][kk])
            if len(runningRTFA_JUV_end[key][kk]) > 0:
                JUVProbeNogoRT = np.append(JUVProbeNogoRT, runningRTFA_JUV_end[key][kk])

        adtColor = (255 / 255, 189 / 255, 53 / 255)
        juvColor = (63 / 255, 167 / 255, 150 / 255)

            # get rid of the nan values
        ADTProbeGoRT = ADTProbeGoRT[~np.isnan(ADTProbeGoRT)]
        ADTProbeNogoRT = ADTProbeNogoRT[~np.isnan(ADTProbeNogoRT)]
        JUVProbeGoRT = JUVProbeGoRT[~np.isnan(JUVProbeGoRT)]
        JUVProbeNogoRT = JUVProbeNogoRT[~np.isnan(JUVProbeNogoRT)]
            # Label the columns of the DataFrame

        aveRT_plot = StartPlots()

                # create a figure and axis object

                # plot the violins
            #sns.violinplot([ADTProbeGoRT, ADTProbeNogoRT], ax = aveRT_plot.ax, color = adtColor)
            #sns.violinplot([JUVProbeGoRT, JUVProbeNogoRT], ax = aveRT_plot.ax, color=juvColor)
        vp1 = aveRT_plot.ax.violinplot([ADTProbeGoRT, ADTProbeNogoRT],
                    positions=[1,4], showmedians=True, showextrema=False, widths=0.5)
        vp2 = aveRT_plot.ax.violinplot([JUVProbeGoRT, JUVProbeNogoRT],
                    positions=[2,5], showmedians=True, showextrema=False, widths=0.5)

        vp1['bodies'][0].set_facecolor(adtColor)
        vp1['bodies'][1].set_facecolor(adtColor)
        vp1['cmedians'].set_color(adtColor)

        vp2['bodies'][0].set_facecolor(juvColor)
        vp2['bodies'][1].set_facecolor(juvColor)
        vp2['cmedians'].set_color(juvColor)


        aveRT_plot.ax.set_xticks([1, 2, 4, 5])
        aveRT_plot.ax.set_xticklabels(['ADT go', 'JUV go', 'ADT nogo', 'JUV nogo'])

        aveRT_plot.ax.set_ylabel('Average response time (s)')
        titletext = '-'.join(str(i) for i in cues)
        aveRT_plot.ax.set_title('Average response time cue(' + titletext + ')')
        aveRT_plot.fig.set_figwidth(8)

        aveRT_plot.save_plot('Average response time (end stage) cue(' + titletext + ').svg', 'svg', save_path)
        aveRT_plot.save_plot('Average response time (end stage) cue(' + titletext + ').tif', 'tif', save_path)
                # perform two way anova
        numData = len(ADTProbeGoRT) + len(ADTProbeNogoRT) + len(JUVProbeGoRT) + len(JUVProbeNogoRT)

        age_anova = []
        response_anova = []
        respT_anova = []

                # loop through every group
        for t in ADTProbeGoRT:
            age_anova.append('ADT')
            response_anova.append('go')
            respT_anova.append(t)

        for t in ADTProbeNogoRT:
            age_anova.append('ADT')
            response_anova.append('nogo')
            respT_anova.append(t)

        for t in JUVProbeGoRT:
            age_anova.append('JUV')
            response_anova.append('go')
            respT_anova.append(t)

        for t in JUVProbeNogoRT:
            age_anova.append('JUV')
            response_anova.append('nogo')
            respT_anova.append(t)

        anova_data = pd.DataFrame({'age':age_anova,
                                           'response': response_anova,
                                           'respT': respT_anova
                                           })
        model = ols('respT ~ age + response + age:response', anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

                # print ANOVA table
        print(anova_table)

        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        mc = pairwise_tukeyhsd(anova_data['respT'], anova_data['age'] + anova_data['response'])
        print(mc)

    def plot_psycho(self, save_path):
        # for different stages, plot the average psychometric curve for adult and juvenile animal separately
        # need to get last 3 sessions for all animals (suppose to be more stable
        wholeSess = np.arange(len(self.trainSti))
        sessionInd =[]
        for aa in np.unique(self.beh_dict['animal']):
            sess = np.array(wholeSess[np.logical_and([True if a==aa else False for a in self.beh_dict['animal']],
                                            self.trainSti>=8)])
            for s in sess[-5:]:
                sessionInd.append(s)

        tempL_fit = self.beh_dict['L-fit'][:,sessionInd]
        tempPsycho_data = self.beh_dict['psycho_data'][:,sessionInd]
        tempAnimal = [self.beh_dict['animal'][i] for i in sessionInd]

        ADTInd = [i for i in range(len(tempAnimal)) if 'ADT' in tempAnimal[i]]
        JUVInd = [i for i in range(len(tempAnimal)) if 'JUV' in tempAnimal[i]]

        probeInd = [1, 3, 5, 7, 8, 10, 12, 14]
        taskInd = [0, 2, 4, 6, 9, 11, 13, 15]


        meanADTChoice = np.nanmean(tempPsycho_data[:,ADTInd],1)
        steADTChoice = np.nanstd(tempPsycho_data[:,ADTInd],1)/np.sqrt(
            np.count_nonzero(~np.isnan(tempPsycho_data[:,ADTInd]), axis=1))

        meanJUVChoice = np.nanmean(tempPsycho_data[:,JUVInd],1)
        steJUVChoice = np.nanstd(tempPsycho_data[:,JUVInd],1)/np.sqrt(
            np.count_nonzero(~np.isnan(tempPsycho_data[:,JUVInd]), axis=1))

        adtColor = (255 / 255, 189 / 255, 53 / 255)
        juvColor = (63 / 255, 167 / 255, 150 / 255)

        psycho_plot = StartPlots()
        psycho_plot.ax.scatter(self.beh_dict['stiFreq'][taskInd],meanADTChoice[taskInd],
                               marker='.', s = 100, c=adtColor,label='_nolegend_')
        psycho_plot.ax.errorbar(self.beh_dict['stiFreq'][taskInd],meanADTChoice[taskInd],
                    yerr=steADTChoice[taskInd], fmt='none', ecolor=adtColor,label='_nolegend_')
        psycho_plot.ax.scatter(self.beh_dict['stiFreq'][probeInd], meanADTChoice[probeInd],
                               marker='s', c=adtColor, label='_nolegend_')
        psycho_plot.ax.errorbar(self.beh_dict['stiFreq'][probeInd], meanADTChoice[probeInd],
                                yerr=steADTChoice[probeInd], fmt='none', ecolor=adtColor, label='_nolegend_')

        psycho_plot.ax.scatter(self.beh_dict['stiFreq'][taskInd],meanJUVChoice[taskInd],
                               marker='.', s = 100, c=juvColor, label='_nolegend_')
        psycho_plot.ax.errorbar(self.beh_dict['stiFreq'][taskInd],meanJUVChoice[taskInd],
                    yerr=steJUVChoice[taskInd], fmt='none', ecolor=juvColor, label='_nolegend_')
        psycho_plot.ax.scatter(self.beh_dict['stiFreq'][probeInd], meanJUVChoice[probeInd],
                               marker='s', c=juvColor, label='_nolegend_')
        psycho_plot.ax.errorbar(self.beh_dict['stiFreq'][probeInd], meanJUVChoice[probeInd],
                                yerr=steJUVChoice[probeInd], fmt='none', ecolor=juvColor, label='_nolegend_')

        # run statistic analysis:
        # t-test
        pVal = np.zeros(tempPsycho_data.shape[0])
        for i in range(tempPsycho_data.shape[0]):
            _, pVal[i] = ttest_ind(tempPsycho_data[i,JUVInd], tempPsycho_data[i,ADTInd], nan_policy='omit')

        # anova
        cue_anova = []
        age_anova = []
        response_anova = []

        # loop through every group
        for t in range(tempPsycho_data.shape[1]):
            for cue in range(tempPsycho_data.shape[0]):
                if t in ADTInd:
                    age_anova.append('ADT')
                elif t in JUVInd:
                    age_anova.append('JUV')
                cue_anova.append(str(cue+1))
                response_anova.append(tempPsycho_data[cue,t])

        anova_data = pd.DataFrame({'age': age_anova,
                                   'cue': cue_anova,
                                   'response': response_anova
                                   })
        model = ols('response ~ age + cue + age:cue', anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # print ANOVA table
        print(anova_table)

        # plot significance
        # for tt in range(tempPsycho_data.shape[0]):
        #     if pVal[tt] < 0.05:
        #         dP_plot.ax.plot(xAxis[tt] + 1 * np.array([-0.5, 0.5]), [4, 4],
        #                         color=(1, 0, 0), linewidth=5)

        # plot the fitted results
        # bootstrap to get the mean parameter of the fitted result

        # plot average choice curve and fitted curve
        # plot average data with standard error and average fitted curve

        fit_x = np.linspace(self.beh_dict['stiFreq'][0], self.beh_dict['stiFreq'][-1], 50)

        midFreq = (self.beh_dict['stiFreq'][6] + self.beh_dict['stiFreq'][9])/2
        x_1 = np.linspace(self.beh_dict['stiFreq'][0], self.beh_dict['stiFreq'][-1], 50)
        x_1[fit_x>midFreq] = 0
        x_2 = np.linspace(self.beh_dict['stiFreq'][0], self.beh_dict['stiFreq'][-1], 50)
        x_2[fit_x<midFreq] = 0

        x_model = np.array([np.ones(len(fit_x)),x_1,x_2])

        X = np.dot(np.transpose(tempL_fit[:, ADTInd]), x_model)
        logit_X = 1/(1+np.exp(-X))
        L_ADT = bootstrap(np.transpose(logit_X), 1, 3, 1000)

        X = np.dot(np.transpose(tempL_fit[:, JUVInd]), x_model)
        logit_X = 1/(1+np.exp(-X))
        L_JUV = bootstrap(np.transpose(logit_X), 1, 3, 1000)

        # plot the fitted curve
        X = L_ADT['bootAve']
        psycho_plot.ax.plot(fit_x, L_ADT['bootAve'], color=adtColor, label='Adult')
        psycho_plot.ax.fill_between(fit_x, L_ADT['bootLow'],
                                L_ADT['bootHigh'], color=adtColor, alpha=0.2, label='_nolegend_')

        psycho_plot.ax.plot(fit_x, L_JUV['bootAve'], color=juvColor, label='Juvenile')
        psycho_plot.ax.fill_between(fit_x, L_JUV['bootLow'],
                                L_JUV['bootHigh'], color=juvColor, alpha=0.2, label='_nolegend_')

        psycho_plot.legend(['Adult', 'Juvenile'])
        psycho_plot.ax.set_xlabel('Stimulus frequency')
        psycho_plot.ax.set_ylabel('Response rate')
        psycho_plot.ax.plot()

        psycho_plot.save_plot('Average psychometric.svg', 'svg', save_path)
        psycho_plot.save_plot('Average psychometric.tif', 'tif', save_path)

    def get_meanste(self, data):
        """
        for a input dictionary, return average and standard error
        used only for deal with response time data
        """
        ave = np.zeros((len(data.keys())))
        ste = np.zeros((len(data.keys())))

        for idx, kk in enumerate(data.keys()):
            ave[idx] = np.nanmean(data[kk])
            ste[idx] = np.nanstd(data[kk]) / np.sqrt(np.count_nonzero(~np.isnan(data[kk])))

        return ave, ste

    def check_rate(self, rate):
        # for d-prime calculation
        # if value == 1, change to 0.9999
        # if value == 0, change to 0.0001
        if rate == 1:
            rate = 0.9999
        elif rate == 0:
            rate = 0.0001

        return rate

if __name__ == "__main__":
    # test single session
    animal = 'JUV015'
    session = '220409'
    input_path = r'Z:\Madeline\processed_data\JUV022\230127\JUV022_230127_behaviorLOG.mat'
    x = GoNogoBehaviorMat(animal, session, input_path)
    x.to_df()
    #
    output_path = r'Z:\HongliWang\Madeline\analysis\behavior\JUV015\220409'
    plot_path = os.path.join(output_path, 'beh_plot')
    # x.beh_cut(plot_path)

    ifrun = True
    x.beh_session(plot_path, ifrun)
    x.psycho_curve(plot_path, ifrun)
    x.response_time(plot_path, ifrun)
    x.lick_rate(plot_path, ifrun)
    x.ITI_distribution(plot_path, ifrun)
    x.running_aligned('onset',plot_path, ifrun)
    # # test code for plot
    #
    x.save_analysis(output_path,ifrun)


    root_dir = r'Z:\HongliWang\Madeline'
    beh_sum = GoNogoBehaviorSum(root_dir)
    # matplotlib.use('Agg')
    beh_sum.process_singleSession(ifrun=True)
    beh_sum.read_data()

    matplotlib.use('QtAgg')
    savefigpath = os.path.join(root_dir, 'summary', 'behavior')
    beh_sum.plot_dP(savefigpath)
    beh_sum.plot_rt([1, 2, 3, 4, 5, 6, 7, 8], savefigpath)
    beh_sum.plot_rt([1, 8], savefigpath)
    beh_sum.plot_rt([2, 7], savefigpath)
    beh_sum.plot_rt([3, 6], savefigpath)
    beh_sum.plot_rt([4, 5], savefigpath)
    beh_sum.plot_rt([9,10,11,12,13,14,15,16], savefigpath)
    beh_sum.plot_psycho(savefigpath)
    matplotlib.use('QtAgg')
    x=1