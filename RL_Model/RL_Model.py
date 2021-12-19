import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

class participant():
    """
    class to keep values of participant, predict selection, and update values
    according to reward for Single Context Experiments

    key attribute:
    self.values, list, in the form of [[V_AL, V_AR], [V_BL, V_BR]]. Only the
    two values related to the input sequence will be used. Based on the mode,
    two or four values will be upgraded.

    self.alphas, learning rate matrix, list, in the form of [[a_AL, a_AR], [a_BL, a_BR]],
    each alpha is a constant and not updated throughout the process.

    self.beta, the linear parameter of stockxxxxx, should be between 0 and 1?

    Since values and alphas are in the form of x_ij, where i/j = 0 for A/L, or 1 for B/R,
    the calculation can be made by using index of list, which, hopefully, will be easier.

    self.mode, 2 or 4, marking if only 2 values will be updated, or all four values will be
    updated, as suggested by different model

    self.values_history, list of all past values, so a list of list (which is of list again)
    """

    def __init__(self, name=None, values=[[1,0],[0.5,0.5]], alphas=[[0.3,0.1],[0.1,0.3]], beta=1.0, mode=2):
        if name is None:
            self.name = "participant"
        else:
            self.name = str(name)

        self.values = values
        #print("Init Values: {v}".format(v = self.values))
        self.values_history = [[[],[]],[[],[]]]
        self.alphas = alphas
        self.beta = beta

        self.possible_sequence = ["A", "B"]
        self.possible_response = ["L", "R"]

        assert mode in [1,2,4], "Unknown mode, set mode to 1, 2 or 4!"
        self.mode = mode

    def getProbabilities(self, sequence):
        # Calculate P_L and P_R from values
        V_L, V_R = self.values[self.possible_sequence.index(sequence)]
        Z = self.beta * (V_L - V_R)
        P_L = 1 / (1 + np.exp(-1 * Z))
        return P_L, 1-P_L

    def makeSelection(self, sequence):
        P_L, P_R = self.getProbabilities(sequence)
        if random.uniform(0, 1) <= P_L:
            return 0, "L"
        else:
            return 1, "R"

    def upgradeValues(self, sequence, response, reward):
        self.values_history[0][0].append(self.values[0][0])
        self.values_history[1][0].append(self.values[1][0])
        self.values_history[1][1].append(self.values[1][1])
        self.values_history[0][1].append(self.values[0][1])

        idx_seq = self.possible_sequence.index(sequence)
        idx_seq_re = abs(1 - idx_seq)
        idx_res = self.possible_response.index(response)
        idx_res_re = abs(1 - idx_res)

        self.values[idx_seq][idx_res] = self.values[idx_seq][idx_res] + \
                                        self.alphas[idx_seq][idx_seq] * \
                                        (reward - self.values[idx_seq][idx_res])

        if self.mode >= 2:
            self.values[idx_seq_re][idx_res] = self.values[idx_seq_re][idx_res] + \
                                        self.alphas[idx_seq][idx_seq_re] * \
                                        (reward - self.values[idx_seq_re][idx_res])

        if self.mode == 4:
            self.values[idx_seq][idx_res_re] = self.values[idx_seq][idx_res_re] + \
                                            self.alphas[idx_seq][idx_seq] * \
                                            (1 - reward - self.values[idx_seq][idx_res_re])

            self.values[idx_seq_re][idx_res_re] = self.values[idx_seq_re][idx_res_re] + \
                                               self.alphas[idx_seq][idx_seq_re] * \
                                               (1 - reward - self.values[idx_seq_re][idx_res_re])


class probabilityCalculator():
    def __init__(self, subject_no=9, mode=2, assist_alpha=True):
        data_filename = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Result/" \
                        "SingleContext_Rat" + str(subject_no) + ".csv"
        self.df = pd.read_csv(data_filename, sep=",")
        subject_name = data_filename.split("_")[-1]
        self.subject_name = subject_name.split(".")[0]
        self.mode=mode
        self.subject_no = subject_no
        self.assist_alpha = assist_alpha

    def _calcEngine(self, alpha_starts = [[0,0],[0,0]], alpha_ends=[[1,1],[1,1]], step=0.1):
        result_df = pd.DataFrame(columns=["Alpha_00","Alpha_01","Alpha_10","Alpha_11","Ln_Likelyhood"])
        name = self.subject_name
        alphas = [[1,2],[3,4]]
        for i in [0,1]:
            for j in [0,1]:
                alphas[i][j] = np.arange(np.max([alpha_starts[i][j],0]),
                                         np.min([1,alpha_ends[i][j]]), step)
        if not self.assist_alpha:
            alphas[0][1] = np.array([0])
            alphas[1][0] = np.array([0])
        print("alphas created")

        for i, alpha_00 in enumerate(alphas[0][0]):
            print("Alpha_00: {v:.3f}, {i}/{j}".format(v=alpha_00, i=i+1, j=len(alphas[0][0])))
            for alpha_01 in alphas[0][1]:
                for alpha_10 in alphas[1][0]:
                    for alpha_11 in alphas[1][1]:
                        alphas2use = [[alpha_00, alpha_01],[alpha_10, alpha_11]]
                        probability_sum = 0.0

                        if self.subject_no <= 16:
                            p = participant(name=name, values=[[1,0],[1,0]],
                                        alphas=alphas2use, beta=1.0, mode=self.mode)
                        else:
                            p = participant(name=name, values=[[0,1],[0,1]],
                                        alphas=alphas2use, beta=1.0, mode=self.mode)

                        #print("Alphas: {a}".format(a=alphas2use))
                        for i in range(len(self.df)):
                            seq_trial = self.df["Sequence"].iloc[i]
                            res_trial = self.df["Response"].iloc[i]
                            reward = self.df["Correct"].iloc[i]
                            if res_trial in ["L", "R"]:
                                prob_trial = p.getProbabilities(seq_trial)
                                probability_sum = probability_sum + np.log(prob_trial[p.possible_response.index(res_trial)])
                                p.upgradeValues(seq_trial, res_trial, reward)

                        df_try = pd.DataFrame([[alpha_00, alpha_01, alpha_10, alpha_11, probability_sum]],
                                              columns=["Alpha_00","Alpha_01","Alpha_10","Alpha_11","Ln_Likelyhood"])
                        result_df = result_df.append(df_try)

        idx = result_df["Ln_Likelyhood"] == result_df["Ln_Likelyhood"].max()
        best_alphas = [[result_df["Alpha_00"].loc[idx].values[0],result_df["Alpha_01"].loc[idx].values[0]],\
                      [result_df["Alpha_10"].loc[idx].values[0],result_df["Alpha_11"].loc[idx].values[0]]]
        return best_alphas, result_df


rootpath = "/media/zhemengwu/Gigantic Data/SingleContextSequence/RL_Model"
if not os.path.exists(rootpath):
    os.mkdir(rootpath)

#for subject_no in np.arange(12,13,1):
subject_all = np.arange(9, 25, 1)
subject_all = subject_all[subject_all != 18]
for subidx, subject_no in enumerate(subject_all):
    if not os.path.exists(os.path.join(rootpath, "Rat_{s}".format(s=subject_no))):
        os.mkdir(os.path.join(rootpath, "Rat_{s}".format(s=subject_no)))
    for mode in [1,2,3,4]:
        print("="*30)
        print("Subject {s}, {i}/15; mode: {m}".format(s=subject_no, i=subidx+1, m=mode))
        if mode == 1:
            Pcalculator = probabilityCalculator(subject_no=subject_no, mode=2, assist_alpha=False)
        elif mode == 3:
            Pcalculator = probabilityCalculator(subject_no=subject_no, mode=4, assist_alpha=False)
        else:
            Pcalculator = probabilityCalculator(subject_no=subject_no, mode=mode, assist_alpha=True)

        alpha_starts = [[0, 0], [0, 0]]
        alpha_ends = [[1, 1], [1, 1]]
        for i, step in enumerate([0.1,0.02,0.004,0.001]):
            print("Cycle {i}...".format(i=i+1))
            best_alphas, result_df = Pcalculator._calcEngine(alpha_starts=alpha_starts, alpha_ends=alpha_ends, step=step)
            print(best_alphas)
            filename = os.path.join(rootpath,
                                "Rat_{s}".format(s=subject_no),
                                "Rat{s}_Mode{m}_Cycle{i}.csv".format(s=subject_no, m=mode, i=i+1))
            with open(filename, "w") as f:
                f.write(result_df.to_csv(index=False))
            alpha_starts = np.array(best_alphas) - step
            alpha_ends = np.array(best_alphas) + step


###################################################33
# Output results
columns = ["Subject", "N_Trials",
           "Mode1_LnLikelyhood", "Mode1_LnLikelyhood_Pertrial",
           "Mode1_Alpha_00","Mode1_Alpha_01", "Mode1_Alpha_10","Mode1_Alpha_11",
           "Mode2_LnLikelyhood", "Mode2_LnLikelyhood_Pertrial",
           "Mode2_Alpha_00","Mode2_Alpha_01", "Mode2_Alpha_10","Mode2_Alpha_11",
           "Mode3_LnLikelyhood", "Mode3_LnLikelyhood_Pertrial",
           "Mode3_Alpha_00","Mode3_Alpha_01", "Mode3_Alpha_10","Mode3_Alpha_11",
           "Mode4_LnLikelyhood", "Mode4_LnLikelyhood_Pertrial",
           "Mode4_Alpha_00","Mode4_Alpha_01", "Mode4_Alpha_10","Mode4_Alpha_11"]
df_sum = pd.DataFrame(columns=columns)

for subject_no in subject_all:
    if not os.path.exists(os.path.join(rootpath, "Rat_{s}".format(s=subject_no))):
        os.mkdir(os.path.join(rootpath, "Rat_{s}".format(s=subject_no)))

    session_df_file = os.path.join("/media/zhemengwu/Gigantic Data/SingleContextSequence",
                                       "Result", "SingleContext_Rat{s}.csv".format(s=subject_no))
    session_df = pd.read_csv(session_df_file, sep=",")
    session_df = session_df.loc[session_df["Response"] != "N"]
    sub_name = "Rat_{s}".format(s=subject_no)
    n_trials = len(session_df)
    values = [sub_name, n_trials]

    for mode in [1,2,3,4]:
        print("=" * 30)
        print("Subject {s}, mode: {m}".format(s=subject_no, m=mode))

        # get mode2use and best-alphas
        if mode in [1,3]:
            mode2use = mode + 1
        else:
            mode2use = mode

        alpha_filename = os.path.join(rootpath,
                                "Rat_{s}".format(s=subject_no),
                                "Rat{s}_Mode{m}_Cycle4.csv".format(s=subject_no, m=mode))
        result_df = pd.read_csv(alpha_filename, sep=",")
        idx = result_df["Ln_Likelyhood"] == result_df["Ln_Likelyhood"].max()
        best_alphas = [[result_df["Alpha_00"].loc[idx].median(), result_df["Alpha_01"].loc[idx].median()], \
                       [result_df["Alpha_10"].loc[idx].median(), result_df["Alpha_11"].loc[idx].median()]]
        # apend values for df_sum
        values.append(result_df["Ln_Likelyhood"].max())
        values.append(result_df["Ln_Likelyhood"].max() / n_trials)
        values.append(result_df["Alpha_00"].loc[idx].median())
        values.append(result_df["Alpha_01"].loc[idx].median())
        values.append(result_df["Alpha_10"].loc[idx].median())
        values.append(result_df["Alpha_11"].loc[idx].median())

        # re-calculate values
        if subject_no <= 16:
            rat = participant(name=sub_name, values=[[1,0],[1,0]],
                            alphas=best_alphas, beta=1.0, mode=mode2use)
        else:
            rat = participant(name=sub_name, values=[[0,1],[0,1]],
                            alphas=best_alphas, beta=1.0, mode=mode2use)
        seqA_X, seqA_Y = [], []
        seqB_X, seqB_Y = [], []
        for i in range(len(session_df)):
            seq_trial = session_df["Sequence"].iloc[i]
            res_trial = session_df["Response"].iloc[i]
            reward = session_df["Correct"].iloc[i]
            if res_trial in ["L", "R"]:
                rat.upgradeValues(seq_trial, res_trial, reward)
                if seq_trial == "A":
                    seqA_X.append(i)
                    seqA_Y.append(reward)
                else:
                    seqB_X.append(i)
                    seqB_Y.append(reward)
        # other parameter for plot
        x_plot = np.arange(len(session_df))

        # Sequence A Values
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax1 = fig.add_subplot(1, 1, 1)
        if subject_no <= 16:
            ax1.plot(x_plot, np.array(rat.values_history[0][0]), "b")
            ax1.plot(x_plot, np.array(rat.values_history[0][1]), "r")
        else:
            ax1.plot(x_plot, np.array(rat.values_history[0][0]), "r")
            ax1.plot(x_plot, np.array(rat.values_history[0][1]), "b")
        colormap = ["r", "b"]
        shapemap = ["x", "o"]
        for x, y in zip(seqA_X, seqA_Y):
            ax1.scatter(x, (y-0.5) * 1.05 + 0.5, color=colormap[y], marker=shapemap[y], s=5)
        ax1.set_xlabel("Sequences A Values", fontsize=12)
        ax1.set_ylim([-0.03,1.03])
        ax1.set_xlim([0, n_trials])
        ax1.grid()
        filename = os.path.join(rootpath, "Rat_{s}".format(s=subject_no), "SequenceA_Mode{m}.png".format(m=mode))
        plt.savefig(filename)
        plt.close(fig)

        # Sequence B Values
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax2 = fig.add_subplot(1, 1, 1)
        if subject_no <= 16:
            ax2.plot(x_plot, np.array(rat.values_history[1][0]), "r")
            ax2.plot(x_plot, np.array(rat.values_history[1][1]), "b")
        else:
            ax2.plot(x_plot, np.array(rat.values_history[1][0]), "b")
            ax2.plot(x_plot, np.array(rat.values_history[1][1]), "r")
        colormap = ["r", "b"]
        shapemap = ["x", "o"]
        for x, y in zip(seqB_X, seqB_Y):
            ax2.scatter(x, (y-0.5) * 1.05 + 0.5, color=colormap[y], marker=shapemap[y], s=5)
        ax2.set_xlabel("Sequences B Values", fontsize=12)
        ax2.set_ylim([-0.03,1.03])
        ax2.set_xlim([0, n_trials])
        ax2.grid()
        filename = os.path.join(rootpath, "Rat_{s}".format(s=subject_no), "SequenceB_Mode{m}.png".format(m=mode))
        plt.savefig(filename)
        plt.close(fig)

    df2add = pd.DataFrame([values], columns=columns)
    df_sum = df_sum.append(df2add, ignore_index=True)

with open(os.path.join(rootpath, "Summary.csv"), "w") as f:
    f.write(df_sum.to_csv(index=False))





