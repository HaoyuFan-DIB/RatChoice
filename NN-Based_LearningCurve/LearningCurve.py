#######################################################
# Model to fit the accuracy curve of both SqA and SqB #
#######################################################
# The fitting process is based on the sherpa package,
# but the model can be easily convert to different
# packages like lmfit etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.optmethods import NelderMead
from sherpa.fit import Fit
from sherpa.data import Data1D


def sigmoid(x, amplitude, delay, scale):
    """
    Function to return sigmoid curve on given x_grid
    y = amplitude / (1 + exp[-(x-delay)/timescale]
    :param x: ndarray, grid to calculate from, usually array of session numbers (1 to 28)
    :param amplitude: float, dynamic range of sigmoid, from -inf to inf
    :param delay: float, shift of the grid
    :param scale: float, time-scale
    :return:
    """
    t = (x - delay) / scale
    y = amplitude / (1 + np.exp(-t))
    return y


class LearningCurve(ArithmeticModel):
    """
    # A sherpa model to simulate the accurate rate of SqA and SqB
    attribute:
    name, str, name of model, default is "LearningCurve"
    single_sigmoid, bool, if True, SqA and SqB share the same learning curve, and vice versa
    parameters, 3 sigmoid curves each of which have 3 parameters, 9 in total. If single_sigmoid
                is True, then only 6 of them is effective.
    methods:
    __init__: required for initialization
    calc: required to evaluate model
    OutputParameter: quick access to the values of parameters
    """

    # combined model to reproduce the accuracy curves of SqA and SqB.
    # The assumptions are as follows:
    # For both SqA and SqB, the accuracy comes from two part: learning new part and remembering old part
    # The leaning part stands for where the rats knows the rule so they made the right responses.
    # The portion of learning curve increases over time as the learning process
    # In the memory part, rats make their responses depending on the ratio of SqA and SqB in their memory.
    # For example, the ratio of SqA is very high at the beginning of experiment.
    # This ratio changes over time and does not have to be 50-50 at a given time point.
    # We also assume the increase in the memory portion and the changes in the ratio follows Sigmoid or
    # reverse sigmoid curve over the time

    # Under this context, the rats have strong preference on SqA at the beginning. This makes them do SqA
    # correctly without understanding the rule. This ratio became less extreme as experiment went on, so
    # while the memory portion in SqA is becoming larger, it is increasing at a lower rate than the decrease
    # in the ratio, which makes the total accuracy rate in SqA decrease

    def __init__(self, name=None, single_sigmoid=True):
        if name is None:
            name = "LearningCurve"
        self.single_sigmoid = single_sigmoid

        # Setup parameters, initial values, and upper/lower boundaries
        # These parameters are attribute of the model, and their values will be changed during the fitting
        # To Do: add boundaries for scale parameters??

        self.pref_amp = Parameter(name, 'pref_amp', 0.5, frozen=False, min=0, max=1)
        self.pref_delay = Parameter(name, 'pref_delay', 5, frozen=False, min=-30, max=60)
        self.pref_scale = Parameter(name, 'pref_scale', 5, frozen=False, min=0.5, max=60)

        self.SqA_amp = Parameter(name, 'SqA_amp', 1, frozen=False, min=0, max=1)
        self.SqA_delay = Parameter(name, 'SqA_delay', 5, frozen=False, min=-30, max=60)
        self.SqA_scale = Parameter(name, 'SqA_scale', 10, frozen=False, min=0.5, max=60)

        self.SqB_amp = Parameter(name, 'SqB_amp', 1, frozen=False, min=0, max=1)
        self.SqB_delay = Parameter(name, 'SqB_delay', 5, frozen=False, min=-30, max=60)
        self.SqB_scale = Parameter(name, 'SqB_scale', 10, frozen=False, min=1, max=60)

        ArithmeticModel.__init__(self, name, (self.pref_amp, self.pref_delay, self.pref_scale,
                                              self.SqA_amp, self.SqA_delay, self.SqA_scale,
                                              self.SqB_amp, self.SqB_delay, self.SqB_scale))

        # if single_sigmoid, freeze unnecessary parameters to be more efficient
        if self.single_sigmoid:
            self.SqB_amp.freeze()
            self.SqB_delay.freeze()
            self.SqB_scale.freeze()


    def calc(self, pars, x, *args, **kwargs):
        # the grid is passed in x and parameters in pars as a tuple
        # We want to fit accurate rate of SqA and SqB at the same time.
        # This is done by combining the two accuracy curves into a single X-Y array.
        # So the input x_grid is like [1,2,...28,1,2,...28]. A bit tricky but manageable...

        # get x_grid and parameters
        x = np.split(x, 2)[0]
        pref_amp, pref_delay, pref_scale, SqA_amp, SqA_delay, SqA_scale, SqB_amp, SqB_delay, SqB_scale = pars

        # memory allocation curve
        s_pref = 1 - sigmoid(x, pref_amp, pref_delay, pref_scale)

        # Learning curve for SqA and SqB
        s_sqa = sigmoid(x, SqA_amp, SqA_delay, SqA_scale)
        if self.single_sigmoid:
            s_sqb = s_sqa
        else:
            s_sqb = sigmoid(x, SqB_amp, SqB_delay, SqB_scale)

        # Calculate accuracy rate, combine, output
        sqa_correct = s_sqa + (1 - s_sqa) * s_pref
        sqb_correct = s_sqb + (1 - s_sqb) * (1 - s_pref)
        combined_correct = np.append(sqa_correct, sqb_correct)

        return combined_correct

    def OutputParameter(self):
        # Shortcut to access the values
        out_list = []
        out_list.append(self.pref_amp.val)
        out_list.append(self.pref_delay.val)
        out_list.append(self.pref_scale.val)
        out_list.append(self.SqA_amp.val)
        out_list.append(self.SqA_delay.val)
        out_list.append(self.SqA_scale.val)
        if not self.single_sigmoid:
            out_list.append(self.SqB_amp.val)
            out_list.append(self.SqB_delay.val)
            out_list.append(self.SqB_scale.val)

        return np.asarray(out_list)


if __name__ == "__main__":
    # Basic set-ups
    Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Group'
    groups = ["dHPC", "Ent", "vHPC"]

    for group in groups:
        # Loading csv data to dataframe
        df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContextData-", ".csv"])))
        columns = df.columns[2::2]
        # Get condition names (since they are not same for different groups)
        conditions = []
        for item in columns:
            item = item.split("-")[-1]
            if item not in conditions:
                conditions.append(item)

        for i, condition in enumerate(conditions):
            # sort data structure, get accuracy rates and error bars
            sqa = df["-".join(["SqA", condition])].to_numpy()
            sqa_error = df["-".join(["SqA", condition, "err"])].to_numpy()
            sqb = df["-".join(["SqB", condition])].to_numpy()
            sqb_error = df["-".join(["SqB", condition, "err"])].to_numpy()

            sqa_error = sqa_error[~np.isnan(sqa)]
            sqa = sqa[~np.isnan(sqa)]
            sqb_error = sqb_error[~np.isnan(sqb)]
            sqb = sqb[~np.isnan(sqb)]

            # Crate data2fit
            X = df["Session"].to_numpy()
            X = X[~np.isnan(X)]
            # data2fit = Data1D(name, X, Y)
            # sherpa requires X and Y to have the same length, so we have to duplicate X
            data2fit = Data1D('data2fit', np.append(X, X), np.append(sqa, sqb))


            # model and fit!
            model2fit = LearningCurve(single_sigmoid=True)
            fit = Fit(data2fit, model2fit, method=NelderMead())
            result = fit.fit()


            # print out fitted parameter
            print("="*20)
            print(",".join([group, condition]))
            print(model2fit)
            print("")


            # Graphic output
            fig = plt.figure(figsize=[8, 14], dpi=200)
            x_plot = np.arange(start=np.min(X), stop=np.max(X), step=0.01)
            # finely sampled x_grid to plot smooth curve
            # again we will duplicate x_plot when evaluate the model

            ## upper for overall fitting ##
            # data points and error bars
            ax = fig.add_subplot(3, 1, 1)
            ax.scatter(X, sqa, color="r")
            for x, y, err in zip(X, sqa, sqa_error):
                ax.plot([x, x], [y+err, y-err], linestyle="--", color="r")

            ax.scatter(X, sqb, color="b")
            for x, y, err in zip(X, sqb, sqb_error):
                ax.plot([x, x], [y+err, y-err], linestyle="--", color="b")

            # fitted model
            y_model = model2fit(np.append(x_plot, x_plot))
            y_sqa, y_sqb = np.split(y_model, 2)
            ax.plot(x_plot, y_sqa, color="r", label="SqA")
            ax.plot(x_plot, y_sqb, color="b", label="SqB")

            # SSE
            SSE = np.sum((np.append(sqa, sqb) - model2fit(np.append(X, X)))**2)
            ax.scatter(np.median(X), np.median(sqa), color="w", label="SSE = %.2f" % SSE)

            # finish up
            ax.grid(color="0.8")
            ax.legend()
            ax.set_ylabel("Accurate Rate")
            ax.set_title("-".join([group, condition]))


            ## lower panels for memory and learning sigmoid ##
            # Prepare data to plot
            para_all = model2fit.OutputParameter()
            sigmoid_mem = 1 - sigmoid(x_plot, para_all[0], para_all[1], para_all[2])
            sigmoid_learn = sigmoid(x_plot, para_all[3], para_all[4], para_all[5])

            # Middle panel for memory
            # Plotting and output parameters
            ax = fig.add_subplot(3, 1, 2)
            ax.plot(x_plot, sigmoid_mem, color="0.5",
                    label="Memory, amp = %.2f, delay = % .2f, time scale = %.2f"
                          % (para_all[0], para_all[1], para_all[2]))
            ax.plot(x_plot, np.ones_like(x_plot)*0.5, color="r", linestyle="--")
            ax.set_ylim([0, 1])
            ax.grid(color="0.8")
            ax.legend(loc="upper right")

            # Lower panel for learning
            ax = fig.add_subplot(3, 1, 3)
            ax.plot(x_plot, sigmoid_learn, color="k",
                    label="Learning , amp = %.2f, delay = % .2f, time scale = %.2f"
                          % (para_all[3], para_all[4], para_all[5]))
            ax.plot(x_plot, np.ones_like(x_plot) * para_all[3], color="orange", linestyle="--")
            ax.set_ylim([0, 1])
            ax.grid(color="0.8")
            ax.legend(loc="upper right")
            ax.set_xlabel("Session (days)")

            # Finish up the entire figure and output
            plt.tight_layout()
            plt.savefig(os.path.join(Rootpath.replace("Group", "LearningCurve"), "_".join([group, condition])+".png"))
            #plt.show()
            plt.close()

