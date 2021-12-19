import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.optmethods import NelderMead
from sherpa.fit import Fit
from sherpa.data import Data1D

# define model for Weibull distribution
class Weibull_cdf(ArithmeticModel):
    # Weibull distribution, Y = A(1 - b^(-[(X/L)^S]))
    # X: Independent variable, time in the form of session no. or date in our case
    # Y: Dependent variable, accuracy in our case
    # b: Base, set upon initialize, e by default (but Gallistel+2004 used "2")
    # A: Asymptote, free parameter between 0 and 1, accuracy to archive in the end (t -> inf)
    # L: Latency, free parameter LT 0, time scale over which Y would change significantly
    # S: Abruptness of onset, free parameter LT 0, significance of the scaled time (X/L)

    def __init__(self, name=None, base=None, reverse=False, A=1.0, lambda_0=2.0, k=2.0, C=0.0, delay=0.0):
        # define parameters of the model
        if name is None:
            name = "Weibull"

        if base is None:
            self.base = np.e
        else:
            self.base = base

        self.reverse = reverse

        self.A = Parameter(name, 'A', A, frozen=False, min=-1, max=1)
        self.lambda_0 = Parameter(name, 'lambda_0', lambda_0, frozen=False, min=0)
        self.k = Parameter(name, 'k', k, frozen=False, min=0)
        self.C = Parameter(name, 'C', C, frozen=False)
        self.delay = Parameter(name, 'Delay', delay, frozen=False)

        ArithmeticModel.__init__(self, name, (self.A, self.lambda_0, self.k, self.C, self.delay))


    def calc(self, pars, x, *args, **kwargs):
        # define how to calculate the model using parameters
        #Y = A(1 - b ^ (-[(X / L) ^ S])) + C
        A, lambda_0, k, C, delay = pars
        if self.reverse:
            x = np.max(x) - x + 1

        power = ((x-delay)/lambda_0) ** k
        return A * (1 - self.base ** (power*-1)) + C

if __name__ == "__main__":
    # loading data
    data_filename = "/media/zhemengwu/Gigantic Data/Weibull/SingleContextData-vHPC.csv"
    df = pd.read_csv(data_filename, sep=",")
    conditions = df.columns[2::2]

    # loop within the four conditions
    for i, condition in enumerate(conditions):
        # Obtain data to be fit (for one condition)
        # use session index, i.e. 1, 2, 3, ..... as independent variable X
        X = df["Session"].to_numpy()
        Y = df[condition].to_numpy()
        Err = df[condition + "-err"].to_numpy()
        data2fit = Data1D('data2fit', X, Y, Err)

        # build model to fit, freeze S to 2.0??
        if "Old" in condition:
            condition = condition.replace("Old", "Old Context")
        if "New" in condition:
            condition = condition.replace("New", "New Context")
        model2fit = Weibull_cdf(name="Weibull-" + condition, base=2)
        model2fit.S.val = 2.0
        model2fit.S.freeze()
        if np.average(Y[0:3]) > np.average(Y[-3:]):
            model2fit.A.val = -0.5
            model2fit.C.val = 1.0

        # model2fit.L.val = 1.0
        # model2fit.L.freeze()
        # model2fit.A.min = 0.0
        model2fit.delay.min = 0.0
        model2fit.delay.freeze()
        # if "SqA" in condition:
        #    model2fit = 1.0 - model2fit

        # fit!
        # optimized by minimizing kai-squares, pts with smaller SEM have more weights!
        fit = Fit(data2fit, model2fit, method=NelderMead())
        result = fit.fit()
        error = fit.est_errors()

        # output result
        # print
        print("=" * 30)
        print("Condition: " + condition)
        try:
            for pname, pval, perr in zip(error.parnames, error.parvals, error.parmaxes):
                print("{name}: {value:.2f} pm {err:.2f}".
                      format(name=pname, value=pval, err=perr))
        except:
            print("Error Estimation Failed...")
            for para in model2fit.pars:
                if not para.frozen:
                    print("{name}: {value:.2f}".format(name=para.fullname, value=para.val))
        print("Reduced Kai Square: {k:.2f}".format(k=result.rstat))
        print("")

        # plot
        plt.scatter(X, Y)
        for x, y, err in zip(X, Y, Err):
            plt.plot([x, x], [y + err, y - err], color="r")
        plt.plot(X, model2fit(X))
        #    plt.xlabel("Sessions\nS={s:.2f} (frozen), A={a:.2f}, C={c:.2f}, L={l:.2f}, delay={d:.2f}, kai2={kai:.2f}".
        #               format(s=model2fit.S.val,
        #                      a=model2fit.A.val,
        #                      c=model2fit.C.val,
        #                      l=model2fit.L.val,
        #                      d=model2fit.delay.val,
        #                      kai=result.rstat))
        plt.xlabel("Sessions\nS={s:.2f} (frozen), A={a:.2f}, C={c:.2f}, L={l:.2f}, kai2={kai:.2f}".
                   format(s=model2fit.S.val,
                          a=model2fit.A.val,
                          c=model2fit.C.val,
                          l=model2fit.L.val,
                          kai=result.rstat))
        plt.ylabel("Accuracy")
        plt.grid()
        plt.title(condition)
        plt.tight_layout()
        plt.savefig(os.path.join("/media/zhemengwu/Gigantic Data/Weibull", condition + ".png"))
        plt.show()
        # plt.close()

        # end of loop
