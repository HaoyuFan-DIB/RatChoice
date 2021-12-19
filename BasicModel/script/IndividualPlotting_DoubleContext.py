# Fitting SqA%, SqB% or SqA+SqB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sherpa.optmethods import NelderMead
from sherpa.fit import Fit
from sherpa.data import Data1D


def parse_column(columns):
    out_dir = {}
    for column in columns:
        sq, roi, rat_id = column.split("-")
        if rat_id not in out_dir.keys():
            out_dir[rat_id] = roi
    return out_dir


def calcAUC(x, y):
    assert len(x) == len(y), "Input x and y must have the same length"
    auc = 0
    idx_l = 0
    while True:
        if idx_l >= len(x) - 1:
            break
        x1, x2 = x[idx_l], x[idx_l + 1]
        y1, y2 = y[idx_l], y[idx_l + 1]
        d_auc = 0.5 * (y1 + y2) * (x2 - x1)
        auc = auc + d_auc
        idx_l = idx_l + 1
    return auc


Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Individual'
groups = ["dHPC", "Ent", "vHPC"]
for group in groups:
    print(group)
    if not os.path.exists(os.path.join(Rootpath, "AUC", group)):
        os.mkdir(os.path.join(Rootpath, "AUC", group))
    df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContext_", ".csv"])))
    rat_dir = parse_column(df.columns[2::])
    X = df["Session"].to_numpy()

    rat_id_all, AUC_all = [], []

    for rat_id in rat_dir.keys():
        # Gather data
        SqA_column = "-".join(["SqA", rat_dir[rat_id], rat_id])
        SqB_column = SqA_column.replace("SqA", "SqB")
        Y_SqA = df[SqA_column].to_numpy()
        Y_SqB = df[SqB_column].to_numpy()

        # remove nan
        idx = (~np.isnan(Y_SqA)) * (~np.isnan(Y_SqB))
        X_rat = X[idx]
        Y_SqA, Y_SqB = Y_SqA[idx], Y_SqB[idx]

        Y_SqA[Y_SqA == 0] = 0.01
        Y_SqB[Y_SqB == 0] = 0.01
        Y = Y_SqA / Y_SqB

        AUC = calcAUC(X_rat, Y - 0.5)   # normalized Y

        # Output result, plot
        label = "AUC: %.2f" % AUC
        plt.scatter(X_rat, Y, color="k")
        plt.plot(X_rat, Y, color="k", label=label)

        plt.grid(color="0.8")
        plt.legend(loc='best')
        plt.xlabel(group)
        #plt.ylim([0, 5])
        plt.savefig(os.path.join(Rootpath, "AUC_Double", group, rat_id + "-" + rat_dir[rat_id] + ".png"))
        #plt.show()
        plt.close()

        # Output result, data
        rat_id_all.append(rat_id)
        AUC_all.append(AUC)

    result_df = pd.DataFrame(list(zip(rat_id_all, AUC_all)), columns=["RatID", "AUC"])
    out_filename = os.path.join(Rootpath, "AUC", "AUCResult_" + group + ".csv")

    with open(out_filename, "w") as f:
        f.write(result_df.to_csv(index=False))