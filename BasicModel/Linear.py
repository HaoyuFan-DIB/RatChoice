import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.optmethods import NelderMead
from sherpa.fit import Fit
from sherpa.data import Data1D

# define model for Linear distribution
class Linear(ArithmeticModel):
    # linear function
    # Y = a * x + b
    # Parameters are thus as follows:
    # X: Independent variable, time in the form of session no. or date in our case
    # Y: Dependent variable, accuracy or inaccuracy in our case
    # a: slope
    # b: intercept

    def __init__(self, name=None, a=0, b=0):
        # define parameters of the model
        if name is None:
            name = "Linear"

        self.a = Parameter(name, 'a', a, frozen=False, min=0)
        self.b = Parameter(name, 'b', b, frozen=False)

        ArithmeticModel.__init__(self, name, (self.a, self.b))


    def calc(self, pars, x, *args, **kwargs):
        # Y = a * x + b
        a, b = pars
        y = a * x + b
        return y


if __name__ == "__main__":
    # loading data
    Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Individual'
    groups = ["dHPC", "Ent", "vHPC"]
    for group in groups:
        df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContextData-", ".csv"])))
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

            Y = sqa + sqb
            Y_err = np.sqrt(sqa_error**2 + sqb_error**2)

            data2fit = Data1D('data2fit', X, Y)
            model2fit = Linear(name="Linear", a=0, b=0)
            fit = Fit(data2fit, model2fit, method=NelderMead())
            result = fit.fit()

            a = model2fit.a.val
            b = model2fit.b.val
            print("="*20)
            print(",".join([group, condition]))
            print(model2fit)
            print("")

            SSE = np.sum((Y - model2fit(X))**2)
            label = condition + ", SSE: %.2f, a: %.2f, b: %.2f" % (SSE, a, b)
            plt.scatter(X, Y, color=colormap[i])
            for x, y, err in zip(X, Y, Y_err):
                plt.plot([x, x], [y + err, y - err], color=colormap[i],linestyle=":")
            plt.plot(X, model2fit(X), color=colormap[i], label=label)

        plt.grid(color="0.8")
        plt.legend()
        plt.xlabel(group)
        plt.show()

