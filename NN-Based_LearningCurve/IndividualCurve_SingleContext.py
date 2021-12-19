# Curve Fitting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from LearningCurve import LearningCurve
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

single_sigmoid = True

Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Individual'
groups = ["dHPC", "Ent", "vHPC"]
for group in groups:
    df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContext_", ".csv"])))
    rat_dir = parse_column(df.columns[2::])
    X = df["Session"].to_numpy()
    X = np.append(X, X)
    X_plot = np.arange(start=np.min(X), stop=np.max(X), step=0.2)
    X_plot = np.append(X_plot, X_plot)

    pic_outdir = os.path.join(Rootpath, group)
    if not os.path.exists(pic_outdir):
        os.mkdir(pic_outdir)

    rat_ID = []
    para_all = []
    SSE = []

    for rat_id in rat_dir.keys():
        # Gather data
        SqA_column = "-".join(["SqA", rat_dir[rat_id], rat_id])
        SqB_column = SqA_column.replace("SqA", "SqB")
        Y_SqA = df[SqA_column].to_numpy()
        Y_SqB = df[SqB_column].to_numpy()
        Y = np.append(Y_SqA, Y_SqB)

        data2fit = Data1D("data2fit", X, Y)
        model2fit = LearningCurve(name=rat_id, single_sigmoid=single_sigmoid)
        fit = Fit(data2fit, model2fit, method=NelderMead())
        result_Sq = fit.fit()

        # Output fit result
        rat_ID.append(rat_id)
        para_all.append(model2fit.OutputParameter())
        SSE.append(np.sum((Y - model2fit(X))**2))
        label = "Overall SSE: %.2f" % (SSE[-1])

        # Plot
        fig = plt.figure(figsize=[8, 6], dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        # SqA
        ax.scatter(np.split(X, 2)[0], np.split(Y, 2)[0], color="r")
        ax.plot(np.split(X_plot, 2)[0], np.split(model2fit(X_plot), 2)[0], color="r", label="SqA")
        # SqB
        ax.scatter(np.split(X, 2)[1], np.split(Y, 2)[1], color="b")
        ax.plot(np.split(X_plot, 2)[1], np.split(model2fit(X_plot), 2)[1], color="b", label="SqB")

        # finish-up
        ax.set_xlabel("-".join([rat_id, rat_dir[rat_id]]) + " Overall SSE: %.2f" % (SSE[-1]))
        ax.grid()
        ax.legend()
        plt.savefig(os.path.join(pic_outdir, "-".join([rat_id, rat_dir[rat_id]])+".png"))
        plt.close()

    para_names = ["amp", "delay", "scale"]
    column_names = ["Pref-" + item for item in para_names]
    if single_sigmoid:
        column_names = column_names + ["Sq-" + item for item in para_names]
    else:
        column_names = column_names + ["SqA-" + item for item in para_names] + ["SqB-" + item for item in para_names]

    result_df = pd.DataFrame(para_all, columns=column_names)
    result_df["RatID"] = rat_ID
    out_filename = os.path.join(Rootpath, "DoubleLearnCurve_" + group + "_Boundary.csv")
    if single_sigmoid:
        out_filename = out_filename.replace("Double", "Single")

    with open(out_filename, "w") as f:
        f.write(result_df.to_csv(index=False))