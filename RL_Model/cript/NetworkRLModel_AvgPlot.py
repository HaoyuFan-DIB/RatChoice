import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import random
import copy
from math import floor
import time

def LRate(x, v):
    """
    function for the asymmetric update
    input x will be amplified by v if x >= 0
    otherwise x is unchanged
    """
    if x >= 0:
        return x * v
    else:
        return x

def guessW(W):
    x = np.arange(-5,5,0.1)
    y = 0.5 + 0.5 * erf(x/np.sqrt(2))

    x_out = x[np.where(y < W)]
    return np.max(x_out)

def printETA(timeremain):
    if timeremain > 86400:
        print("More than {n} days remain...".format(n=floor(timeremain/86400)))
    elif timeremain > 3600:
        print("About {n:.1f} hours remain...".format(n=timeremain/3600))
    elif timeremain > 600:
        print("Less than 1 hour remains...")
    elif timeremain > 60:
        print("Almost there, about {n:.0f} minutes...".format(n=timeremain/60))
    elif timeremain > 1:
        print("In 1 minute...")
    else:
        print("Finished!")

class Participant():
    def __init__(self,
                 alpha = 0.001,  # learning rate
                 sigma = 0.6195, # sigma^-1
                 c=0,            # diffuse parameter
                 v=6.0,          # asymmetry amplifier
                 W_L = [0.7, 0.001, 0.001],    # initial link to E, [port, S_R, S_L]
                 W_R = [0.01, 0.001, 0.001],   # initial link to I, [port, S_R, S_L]
                 asy = 1.0,   # change input x to [1,as,0] or [1,0,as]?
                 use_multi = True):  # if True, use multiplicative model, otherwise use additive model
        self.alpha = alpha
        self.sigma = sigma
        self.c = c
        self.v = v
        self.W_R = np.array(W_R)
        self.W_R_all = [np.array(W_R)]
        self.W_L = np.array(W_L)
        self.W_L_all = [np.array(W_L)]
        self.trials_alls = [0]   # an array that records idx of trials when updates happens
        self.use_multi = use_multi
        self.asy = asy

        # archriving Ws anr accuracy
        self.W_L_Port = [self.W_L[0]]
        self.W_L_L = [self.W_L[1]]
        self.W_L_R = [self.W_L[2]]
        self.W_R_Port = [self.W_R[0]]
        self.W_R_L = [self.W_R[1]]
        self.W_R_R = [self.W_R[2]]

        Corr_left_0 = self.getProbabilities(x=[1,1,0])[0]
        Corr_right_0 = self.getProbabilities(x=[1,0,1])[1]
        self.Corr_Left = [Corr_left_0]
        self.Corr_Right = [Corr_right_0]
        self.Corr_sum = [0.5*(Corr_left_0 + Corr_right_0)]

    def internalQ(self, x):
        # calculate (W_L - W_R) . x, where x = [1,1,0] for left or [1,0,1] for right sound
        # if out > 0, more likely to chose LEFT
        Q = np.dot(np.array(x), self.W_L - self.W_R)
        return Q

    def getProbabilities(self, x=[1,1,0]):
        Q = self.internalQ(x=x)
        P_L = 0.5 + 0.5 * erf(Q / np.sqrt(2))
        return [P_L, 1-P_L]

    def makeChoise(self, x=[1,1,0]):
        P_L, P_R = self.getProbabilities(x=x)
        a = random.uniform(0,1)
        if a <= P_L:
            return [1, 0]  # LEFT
        else:
            return [0, 1]  # RIGHT

    def __getAlphas(self):
        if self.use_multi:
            alpha_L = self.alpha * self.W_L
            alpha_R = self.alpha * self.W_R
        else:
            alpha_L = self.alpha * np.ones(3)
            alpha_R = self.alpha * np.ones(3)

        return alpha_L, alpha_R

    def updateWs(self, x=[1,1,0], Response=[1,0], status=False):
        Q = self.internalQ(x=x)
        alpha_L, alpha_R = self.__getAlphas()
        #assert abs(self.sigma * Q) <= 1.0, "sigma*Q outside [-1,1]?"

        if x[2] == 0:  # x = x1, should choose left
            dW_L = alpha_L * LRate(1 - self.sigma * Q, self.v) * Response[0] * np.array(x) + \
                   self.c * (self.W_R - self.W_L)
            dW_R = alpha_R * LRate(-1 - self.sigma * Q, self.v) * Response[1] * np.array(x) + \
                   self.c * (self.W_L - self.W_R)

        if x[1] == 0: # x = x2, should choose right
            dW_L = alpha_L * LRate(-1 + self.sigma * Q, self.v) * Response[0] * np.array(x) + \
                   self.c * (self.W_R - self.W_L)
            dW_R = alpha_R * LRate(1 + self.sigma * Q, self.v) * Response[1] * np.array(x) + \
                   self.c * (self.W_L - self.W_R)

        if status:
            print("Q is {q:.3f}".format(q=Q))
            print("sigma*Q is {q:.3f}".format(q=Q*self.sigma))
            print("dW_L: {a}".format(a=dW_L))
            print("W_L: {a} --> {b}".format(a=self.W_L, b=self.W_L + dW_L))
            print("dW_R: {a}".format(a=dW_R))
            print("W_R: {a} --> {b}".format(a=self.W_R, b=self.W_R + dW_R))

        self.W_L = self.W_L + dW_L
        self.W_R = self.W_R + dW_R

        self.W_R_all.append(self.W_R)
        self.W_L_all.append(self.W_L)
        self.trials_alls.append(self.trials_alls[-1] + 1)

        self.W_L_Port.append(self.W_L[0])
        self.W_L_L.append(self.W_L[1])
        self.W_L_R.append(self.W_L[2])
        self.W_R_Port.append(self.W_R[0])
        self.W_R_L.append(self.W_R[1])
        self.W_R_R.append(self.W_R[2])

        Corr_left = self.getProbabilities(x=[1, 1, 0])[0]
        Corr_right = self.getProbabilities(x=[1, 0, 1])[1]
        self.Corr_Left.append(Corr_left)
        self.Corr_Right.append(Corr_right)
        self.Corr_sum.append(0.5 * (Corr_left + Corr_right))

        if status:
            print("Accurate rate after updates:")
            print("Left: {l:.3f}; Right: {r:.3f}".format(l=Corr_left, r=Corr_right))

    def evolve(self, bin_trials = 80, n_bins = 28):
        for i in range(n_bins):
            x1, x2 = np.array([1,1,0]), np.array([1,0,1])
            Q1, Q2 = self.internalQ(x=x1), self.internalQ(x=x2)
            Response1, Response2 = self.getProbabilities(x=x1), self.getProbabilities(x=x2)
            alpha_L, alpha_R = self.__getAlphas()

            dW_L_1 = 0.5 * alpha_L * LRate(1 - self.sigma * Q1, self.v) * Response1[0] * np.array(x1)
            # chances for x=x1 is 50%, and P(L|x=x1) is given by Response1[0]
            # under such condition, dW_L will be changed by
            # dW_L = alpha_L * LRate(1 - self.sigma * Q1, self.v) * np.array(x1)
            dW_L_2 = 0.5 * alpha_L * LRate(-1 + self.sigma * Q2, self.v) * Response2[0] * np.array(x2)
            dW_L_diff = self.c * (self.W_R - self.W_L)
            dW_L = dW_L_1 + dW_L_2 + dW_L_diff

            dW_R_1 = 0.5 * alpha_R * LRate(-1 - self.sigma * Q1, self.v) * Response1[1] * np.array(x1)
            dW_R_2 = 0.5 * alpha_R * LRate(1 + self.sigma * Q2, self.v) * Response2[1] * np.array(x2)
            dW_R_diff = self.c * (self.W_L - self.W_R)
            dW_R = dW_R_1 + dW_R_2 + dW_R_diff

            self.trials_alls.append(self.trials_alls[-1] + bin_trials)

            self.W_L = self.W_L + dW_L * bin_trials
            self.W_L_all.append(self.W_L)
            self.W_L_Port.append(self.W_L[0])
            self.W_L_L.append(self.W_L[1])
            self.W_L_R.append(self.W_L[2])

            self.W_R = self.W_R + dW_R * bin_trials
            self.W_R_all.append(self.W_R)
            self.W_R_Port.append(self.W_R[0])
            self.W_R_L.append(self.W_R[1])
            self.W_R_R.append(self.W_R[2])

            Corr_left = self.getProbabilities(x=[1, 1, 0])[0]
            Corr_right = self.getProbabilities(x=[1, 0, 1])[1]
            self.Corr_Left.append(Corr_left)
            self.Corr_Right.append(Corr_right)
            self.Corr_sum.append(0.5 * (Corr_left + Corr_right))

class ExpData():
    def __init__(self, data_path, RatID=11):
        #data_path = "/Users/haoyufan/Sunny/SingleContextSequence/Data"
        #data_path = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Data/"
        data_file = os.path.join(data_path, "SingleContext_Rat{n}.csv".format(n=RatID))
        self.df = pd.read_csv(data_file)
        self.dates = self.df["Date"].unique()

    def getRLCorr(self, n_bins=1):
        R_corr, L_corr = np.array([]), np.array([])
        for date in self.dates:
            df_day = self.df.loc[self.df["Date"] == date]

            df_L = df_day.loc[df_day["Sequence"] == "A"]
            L_corr_day = df_L["Correct"].to_numpy()
            L_corr_day = L_corr_day[L_corr_day != -99]
            L_corr_day = np.array_split(L_corr_day, n_bins)
            L_corr = np.append(L_corr, [np.mean(item) for item in L_corr_day])

            df_R = df_day.loc[df_day["Sequence"] == "B"]
            R_corr_day = df_R["Correct"].to_numpy()
            R_corr_day = R_corr_day[R_corr_day != -99]
            R_corr_day = np.array_split(R_corr_day, n_bins)
            R_corr = np.append(R_corr, [np.mean(item) for item in R_corr_day])

        return L_corr, R_corr

    def getSumCorr(self, n_bins=2):
        Sum_Corr = np.array([])
        for date in self.dates:
            df_day = self.df.loc[self.df["Date"] == date]

            corr_day = df_day["Correct"].to_numpy()
            corr_day = np.array_split(corr_day, n_bins)
            Sum_Corr = np.append(Sum_Corr, [np.mean(item) for item in corr_day])

        return Sum_Corr

class Fitter():
    def __init__(self, data_path):
        # Exp data
        self.ratID = int(input("Rat ID: "))
        expdata = ExpData(data_path, RatID=self.ratID)
        self.L_Corr_exp, self.R_Corr_exp = expdata.getRLCorr(n_bins=1)
        
        # Auto or manual parameter init?
        self.multi = True
        while True:
            respond = input("Do you want to use initial parameter combination?[Y/N]?")
            if respond.upper() in ["Y", "N"]:
                break
        if respond.upper() == "Y":
            self.__autoParameter()
        else:
            self.__manualParameter()
    def __autoParameter(self):
        self.alpha_array = np.arange(start=0.001, stop=0.012, step=0.002)
        self.sigma_array = np.arange(start=0.7, stop=1.31, step=0.05)
        self.v_array = np.arange(start=1., stop=3.1, step=0.3333)
        self.dW = np.arange(start=0.8, stop=1.51, step=0.1)
        self.dW_portion = np.arange(start=0, stop=0.31, step=0.1)
        self.W_port_R = np.arange(start=0, stop=5.1, step=0.5)
        self.W_other = np.arange(start=0, stop=0.51, step=0.1)
        
    def __manualParameter(self):
        self.alpha_array = self.__initParameter("alpha")
        self.sigma_array = self.__initParameter("sigma")

        # asymmetric
        while True:
            respond = input("Include asymmetric prediction error?[Y/N]")
            if respond.upper() in ["Y", "N"]:
                break
        if respond.upper() == "Y":
            self.v_array = self.__initParameter("v")
        else:
            self.v_array = np.array([1.0])

        # W-network
        self.dW = self.__initParameter("dW")
        self.dW_portion = self.__initParameter("dW_portion")
        self.W_port_R = self.__initParameter("W_port_R")
        self.W_other = self.__initParameter("W_other")

    def __initParameter(self, parameter):
        print("="*30)
        start = float(input("Start point of " + parameter.upper() + " array: "))
        stop = float(input("End point of " + parameter.upper() + " array: "))
        if stop > start:
            step = float(input("Step size of " + parameter.upper() + " array: "))
            return np.arange(start=start, stop=stop+0.1*step, step=step)
        else:
            return np.array([start])

    def fit(self):
        best_DF = pd.DataFrame(columns=["Alpha", "Sigma", "v", "dW", "dW_portion", "W_Port_R", "W_other", "SSE"])
        starttime = time.time()
        batch_finished = 0
        batch_tot = len(self.alpha_array)*len(self.sigma_array)*len(self.v_array)
        for alpha in self.alpha_array:
            for sigma in self.sigma_array:
                for v in self.v_array:
                    print("Fitting {m} / {n} batches...".format(m=batch_finished+1, n=batch_tot))
                    tmp_DF = self.__fitSingleBatch(alpha, sigma, v)
                    best_DF = best_DF.append(tmp_DF)
                    best_DF = best_DF.sort_values(by=['SSE'], ascending=True) #, ignore_index=True)
                    idx_max = np.min([200, len(best_DF)])
                    best_DF = best_DF.iloc[0:idx_max]
                    best_DF.to_csv(os.path.join('/psyhome9/wuzhemen/Data/SingleContext_Hipp/FittedResult', 
                                                "Rat{n}_Result_tmp.csv".format(n=self.ratID)), index=False)
                    batch_finished = batch_finished + 1
                    self.__timeModeule(starttime, batch_finished, batch_tot)
        return best_DF

    def __fitSingleBatch(self, alpha, sigma, v):
        tmp_DF = pd.DataFrame(columns=["Alpha", "Sigma", "v", "dW", "dW_portion", "W_Port_R", "W_other", "SSE"])
        for dW in self.dW:
            for dW_portion in self.dW_portion:
                for W_port_R in self.W_port_R:
                    W_port_L = W_port_R + (1 - dW_portion) * dW
                    for W_other in self.W_other:
                        W_SA_L = W_other + dW_portion * dW
                        W_L = [W_port_L, W_SA_L, W_other]
                        W_R = [W_port_R, W_other, W_other]
                        rat = Participant(alpha=alpha, c=0, v=v, sigma=sigma,
                                          W_L=W_L, W_R=W_R, use_multi=self.multi)
                        try:
                            rat.evolve(bin_trials=1, n_bins=28*80)
                        except:
                            continue
                        if np.isnan(rat.Corr_Left).any() or np.isnan(rat.Corr_Right).any():
                            continue

                        L_Corr = np.array_split(rat.Corr_Left[0:-1], 28)
                        L_Corr = np.array([np.mean(item) for item in L_Corr])
                        R_Corr = np.array_split(rat.Corr_Right[0:-1], 28)
                        R_Corr = np.array([np.mean(item) for item in R_Corr])
                        SSE = np.sum(np.abs(L_Corr - self.L_Corr_exp) + np.abs(R_Corr - self.R_Corr_exp))

                        result_slice = {"Alpha": alpha, "Sigma": sigma, "v": v,
                                        "dW": dW, "dW_portion": dW_portion,
                                        "W_Port_R": W_port_R, "W_other": W_other,
                                        "SSE": SSE}
                        tmp_DF = tmp_DF.append(result_slice, ignore_index=True)

        return tmp_DF

    def __timeModeule(self, starttime, finished, total):
        print("{n:.2f}'%' finished...".format(n=finished/total*100))
        time_passed = time.time() - starttime
        time_remain = time_passed / finished * (total-finished)
        printETA(timeremain=time_remain)


if __name__ == "__main__":
    data_path = '/media/zhemengwu/Gigantic Data/SingleContextSequence/FittedResult'
    EXP_datapath = '/media/zhemengwu/Gigantic Data/SingleContextSequence/Data'
    plot_path = '/media/zhemengwu/Gigantic Data/SingleContextSequence/AvgPlot'
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    name_label = "Ent_mcherry_SAL"
    ratID_all = [11,13,16,19,21,24]
    sqrt_n = np.sqrt(len(ratID_all))

    L_exp_all, R_exp_all = [], []
    participant_all = []

    for ratID in ratID_all:
        expdata = ExpData(EXP_datapath, RatID=ratID)
        L_Corr_exp, R_Corr_exp = expdata.getRLCorr(n_bins=1)
        L_exp_all.append(L_Corr_exp)
        R_exp_all.append(R_Corr_exp)

        best_DF = pd.read_csv(os.path.join(data_path, "Rat{n}_Result.csv".format(n=ratID)))
        W_port_L = best_DF["W_Port_R"].iloc[0] + (1 - best_DF["dW_portion"].iloc[0]) * best_DF["dW"].iloc[0]
        W_SA_L = best_DF["W_other"].iloc[0] + best_DF["dW_portion"].iloc[0] * best_DF["dW"].iloc[0]
        best_rat = Participant(alpha=best_DF["Alpha"].iloc[0],
                               sigma=best_DF["Sigma"].iloc[0],
                               v=best_DF["v"].iloc[0],
                               W_L=[W_port_L, W_SA_L, best_DF["W_other"].iloc[0]],
                               W_R=[best_DF["W_Port_R"].iloc[0], best_DF["W_other"].iloc[0],
                                    best_DF["W_other"].iloc[0]],
                               use_multi=True)
        best_rat.evolve(bin_trials=1, n_bins=28 * 80)
        participant_all.append(best_rat)

    # plot1, average accuracy
    # observed data
    x = np.arange(28)*80 + 40
    L_exp_all, R_exp_all = np.asarray(L_exp_all), np.asarray(R_exp_all)
    #print(x.shape)
    #print(np.average(L_exp_all, axis=0).shape)
    plt.plot(x, np.average(L_exp_all, axis=0), marker=".", linestyle="--", color="r", label="Seq A Exp")
    plt.plot(x, np.average(R_exp_all, axis=0), marker=".", linestyle="--", color="b", label="Seq B Exp")
    for i in range(28):
        plt.plot([x[i],x[i]],
                 [np.average(L_exp_all, axis=0)[i] + np.std(L_exp_all, axis=0)[i]/sqrt_n,
                  np.average(L_exp_all, axis=0)[i] - np.std(L_exp_all, axis=0)[i]/sqrt_n],
                 linestyle="--", color="r")
        plt.plot([x[i], x[i]],
                 [np.average(R_exp_all, axis=0)[i] + np.std(R_exp_all, axis=0)[i]/sqrt_n,
                  np.average(R_exp_all, axis=0)[i] - np.std(R_exp_all, axis=0)[i]/sqrt_n],
                 linestyle="--", color="b")
    # model data
    L_model_all = np.asarray([participant_all[ii].Corr_Left[0:-1] for ii in range(len(ratID_all))])
    L_model_all = np.array_split(L_model_all, 28, axis=1) # split into 28 sessions
    #print(np.average(L_model_all[27],axis=1))
    L_model_all = [np.average(L_model_all[ii], axis=1) for ii in range(28)] # average within 28 session

    R_model_all = np.asarray([participant_all[ii].Corr_Right[0:-1] for ii in range(len(ratID_all))])
    R_model_all = np.array_split(R_model_all, 28, axis=1)  # split into 28 sessions
    R_model_all = [np.average(R_model_all[ii], axis=1) for ii in range(28)]  # average within 28 session

    plt.plot(x, np.asarray([np.average(L_model_all[ii]) for ii in range(28)]),
             marker="*", color="r", label="Seq A Model")
    plt.plot(x, np.asarray([np.average(R_model_all[ii]) for ii in range(28)]),
             marker="*", color="b", label="Seq B Model")
    for i in range(28):
        plt.plot([x[i]]*2,
                 [np.average(L_model_all[i])+np.std(L_model_all[i])/sqrt_n,
                  np.average(L_model_all[i])-np.std(L_model_all[i])/sqrt_n],
                 color="r")
        plt.plot([x[i]] * 2,
                 [np.average(R_model_all[i]) + np.std(R_model_all[i])/sqrt_n,
                  np.average(R_model_all[i]) - np.std(R_model_all[i])/sqrt_n],
                 color="b")

    # finish-up
    plt.xlabel("Session")
    plt.ylabel("Correct Rate")
    plt.legend()
    plt.grid(color="0.5")
    plt.savefig(os.path.join(plot_path, name_label+"_AvgAccuracy.png"))
    #plt.show()
    plt.close()


    # plot 2 evolve of W network
    x = np.arange(28*80)
    fig = plt.figure(figsize=(8, 10), dpi=200)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    W_L_all = np.asarray([participant_all[ii].W_L_all for ii in range(len(ratID_all))])
    W_L_all = np.average(W_L_all, axis=0).T
    W_R_all = np.asarray([participant_all[ii].W_R_all for ii in range(len(ratID_all))])
    W_R_all = np.average(W_R_all, axis=0).T
    dW_all = W_L_all - W_R_all

    # ax1, W_L
    ax1.plot(x, W_L_all[0][0:2240], label="Port --> L")
    ax1.plot(x, W_L_all[1][0:2240], label="SqA --> L")
    ax1.plot(x, W_L_all[2][0:2240], label="SqB --> L")
    ax1.legend()
    ax1.grid()
    ylim1 = ax1.get_ylim()

    # ax2, W_R
    ax2.plot(x, W_R_all[0][0:2240], label="Port --> R")
    ax2.plot(x, W_R_all[1][0:2240], label="SqA --> R")
    ax2.plot(x, W_R_all[2][0:2240], label="SqB --> R")
    ax2.legend()
    ax2.grid()
    ylim2 = ax2.get_ylim()

    # make ax2 and ax2 share ylim
    ylim2use = (np.min([ylim1[0], ylim2[0]]), np.max([ylim1[1], ylim2[1]]))
    ax1.set_ylim(ylim2use)
    ax2.set_ylim(ylim2use)

    # ax3, W_L - W_R
    ax3.plot(x, W_L_all[0][0:2240] - W_R_all[0][0:2240], label="d_Port")
    ax3.plot(x, W_L_all[1][0:2240] - W_R_all[1][0:2240], label="d_SqA")
    ax3.plot(x, W_L_all[2][0:2240] - W_R_all[2][0:2240], label="d_SqB")
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel(name_label + " Average Network")

    plt.savefig(os.path.join(plot_path, name_label + "_Network.png"))
    plt.close()
