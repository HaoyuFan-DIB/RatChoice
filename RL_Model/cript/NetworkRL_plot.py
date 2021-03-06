import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import random


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
    def __init__(self, RatID=11):
        #data_path = "/Users/haoyufan/Sunny/SingleContextSequence/Result"
        data_path = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Result/"
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

rat_ID = 16
alpha = 0.004
sigma = 1.06
v = 3.
W_L = [10.86, 0.01, 0.01]
W_R = [9.67, 0.01, 0.01]
rat = Participant(alpha=alpha, c=0, v=v, sigma=sigma, W_L=W_L, W_R=W_R, use_multi=True)
rat.evolve(bin_trials=1, n_bins=28*80)

x = np.arange(28*80)
W_L_all = np.array(rat.W_L_all).T
W_R_all = np.array(rat.W_R_all).T
dW_all = W_L_all - W_R_all

fig = plt.figure(figsize=(8,10), dpi=200)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,1,2)

ax1.plot(x, W_L_all[0][0:2240], label="Port --> L")
ax1.plot(x, W_L_all[1][0:2240], label="SqA --> L")
ax1.plot(x, W_L_all[2][0:2240], label="SqA --> R")
ax1.legend()
ax1.grid()
ylim1 = ax1.get_ylim()

ax2.plot(x, W_R_all[0][0:2240], label="Port --> R")
ax2.plot(x, W_R_all[1][0:2240], label="SqA --> R")
ax2.plot(x, W_R_all[2][0:2240], label="SqB --> R")
ax2.legend()
ax2.grid()
ylim2 = ax2.get_ylim()

ylim2use = (np.min([ylim1[0], ylim2[0]]), np.max([ylim1[1], ylim2[1]]))
ax1.set_ylim(ylim2use)
ax2.set_ylim(ylim2use)

ax3.plot(x, W_L_all[0][0:2240] - W_R_all[0][0:2240], label="d_Port")
print(np.min(W_L_all[0][0:2240] - W_R_all[0][0:2240]))
ax3.plot(x, W_L_all[1][0:2240] - W_R_all[1][0:2240], label="d_L")
ax3.plot(x, W_L_all[2][0:2240] - W_R_all[2][0:2240], label="d_R")
ax3.legend()
ax3.grid()
ax3.set_xlabel("Rat {id}".format(id=rat_ID))

print("sub{id}".format(id=rat_ID))
plt.savefig("/home/zhemengwu/Desktop/Weight_sub{id}_test.png".format(id=rat_ID))


