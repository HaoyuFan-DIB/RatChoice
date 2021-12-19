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
class Plot_RawData(ArithmeticModel):
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
            Y = sqb
            Y_err = sqb_error

            label = condition
            plt.scatter(X, Y, color=colormap[i])
            for x, y, err in zip(X, Y, Y_err):
                plt.plot([x,x], [y+err, y-err], color=colormap[i], linestyle=":")
            plt.plot(X, Y, color=colormap[i], label=label)

        plt.grid(color="0.8")
        plt.legend(loc='best')
        plt.xlabel(group)
        plt.ylim([0, 1])
        plt.show()

