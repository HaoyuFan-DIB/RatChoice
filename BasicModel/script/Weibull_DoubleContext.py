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
# this function is for double context data
class Weibull_DoubleContext(ArithmeticModel):
    # PDF of Weibull distribution
    # Y = (k/lambda) * (x/lambda)^(k-1) * exp[-(x/lambda)^k] (x >= 0)
    # Y = 0   (x < 0)

    # We do not require normalization, so we need to add a scaling factor A to the distribution
    # Parameters are thus as follows:
    # X: Independent variable, time in the form of session no. or date in our case
    # Y: Dependent variable, accuracy or inaccuracy in our case
    # A: Scaling factor for y, positive
    # k: shape parameter, positive
    # lambda: Scaling factor for x, like time-scale, positive

    def __init__(self, name=None, A=1.0, k=5.0, lambda_0=1.0, delay=0):
        # define parameters of the model
        if name is None:
            name = "Weibull"

        self.A = Parameter(name, 'A', A, frozen=False, min=0)
        self.k = Parameter(name, 'k', k, frozen=False, min=0)
        self.lambda_0 = Parameter(name, 'lambda_0', lambda_0, frozen=False, min=0)
        self.delay = Parameter(name, 'delay', delay, frozen=False)
        #self.delay = Parameter(name, 'Delay', delay, frozen=False)

        ArithmeticModel.__init__(self, name, (self.A, self.k, self.lambda_0, self.delay))


    def calc(self, pars, x, *args, **kwargs):
        A, k, lambda_0, delay = pars
        t = np.abs((x - delay) / lambda_0)
        y = A * (k / lambda_0) * t ** (k-1) * np.exp(-1 * t**k)
        return y

if __name__ == "__main__":
    # loading data
    Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Group'
    groups = ["dHPC", "Ent", "vHPC"]
    for group in groups:
        df = pd.read_csv(os.path.join(Rootpath, group.join(["DoubleOldContextData-", ".csv"])))
        columns = df.columns[2::2]
        conditions = []
        for item in columns:
            item = item.split("-")[-1]
            if item not in conditions:
                conditions.append(item)

        colormap = ["r", "b"]
        for i, condition in enumerate(conditions):
            sqa = df["-".join(["SqA", condition])].to_numpy()
            sqa_error = df["-".join(["SqA", condition, "err"])].to_numpy()
            sqb = df["-".join(["SqB", condition])].to_numpy()
            sqb_error = df["-".join(["SqB", condition, "err"])].to_numpy()

            sqa_error = sqa_error[~np.isnan(sqa)]
            sqa = sqa[~np.isnan(sqa)]
            sqb_error = sqb_error[~np.isnan(sqb)]
            sqb = sqb[~np.isnan(sqb)]

            X = df["Session"].to_numpy()
            X = X[~np.isnan(X)]
            Y = sqa / sqb
            Y_err = np.sqrt(sqa_error**2 / sqb**2 + (sqa*sqb_error)**2/sqb**2)

            AUC = np.sum(Y-1) - 0.5*Y[0] - 0.5*Y[-1] + 1
            data2fit = Data1D('data2fit', X, Y)
            model2fit = Weibull_DoubleContext(name="Weibull", A=1.0, k=5.0, lambda_0=1.0, delay=0)
            fit = Fit(data2fit, model2fit, method=NelderMead())
            result = fit.fit()

            A = model2fit.A.val
            k = model2fit.k.val
            lambda_0 = model2fit.lambda_0.val
            delay = model2fit.delay.val
            print("="*20)
            print(",".join([group, condition]))
            print(model2fit)
            print("")

            SSE = np.sum((Y - model2fit(X))**2)
            #label = condition + ", SSE: %.2f, A: %.2f, k: %.2f, \n lambda_0: %.2f, delay: %.2f" % (SSE, A, k, lambda_0, delay)
            label = condition + ", AUC: %.2f" % AUC
            plt.scatter(X, Y, color=colormap[i])
            for x, y, err in zip(X, Y, Y_err):
                plt.plot([x,x], [y+err, y-err], color=colormap[i], linestyle=":")
            plt.plot(X, Y, color=colormap[i], label=label)
            #plt.plot(X, model2fit(X), color=colormap[i], label=label)

        plt.grid(color="0.8")
        plt.legend(loc='best')
        plt.xlabel(group)
        plt.ylim([0, 5])
        plt.show()

