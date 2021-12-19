# Fitting SqA%, SqB% or SqA+SqB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from Sigmoid import Sigmoid
from Linear import Linear
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

#method = "linear"
method = "sigmoid"

Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Individual'
groups = ["dHPC", "Ent", "vHPC"]
for group in groups:
    df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContext_", ".csv"])))
    rat_dir = parse_column(df.columns[2::])
    X = df["Session"].to_numpy()
    X_plot = np.arange(start=np.min(X), stop=np.max(X), step=0.2)

    pic_outdir = os.path.join(Rootpath, group)
    if not os.path.exists(pic_outdir):
        os.mkdir(pic_outdir)

    rat_ID = []
    # parameters for SqA
    a, b, SSE = [], [], []
    if method == "sigmoid":
        c = []

    for rat_id in rat_dir.keys():
        # Gather data
        SqA_column = "-".join(["SqA", rat_dir[rat_id], rat_id])
        SqB_column = SqA_column.replace("SqA", "SqB")
        Y_SqA = df[SqA_column].to_numpy()
        Y_SqB = df[SqB_column].to_numpy()

        # SqB/(SqA + SqB) Model: Sigmoid
        if method == "sigmoid":
            Y = Y_SqB
            # Y = Y_SqA / (Y_SqA + Y_SqB)
            Sq_Model = Sigmoid(name="Sigmoid", a=1.0, b=0, c=1.0)
        if method == "linear":
            Y = Y_SqA + Y_SqB
            Sq_Model = Linear(name="Linear")

        Sq_Data = Data1D("Data", X, Y)
        fit = Fit(Sq_Data, Sq_Model, method=NelderMead())
        result_Sq = fit.fit()

        # Output fit result
        rat_ID.append(rat_id)

        a.append(Sq_Model.a.val)
        b.append(Sq_Model.b.val)
        SSE.append(np.sum((Y - Sq_Model(X))**2))
        #label ="SqA + SqB , SSE: %.2f, a: %.2f, b: %.2f" % (SSE[-1], a[-1], b[-1])
        label = "SqB , SSE: %.2f, a: %.2f, b: %.2f" % (SSE[-1], a[-1], b[-1])
        if method == "sigmoid":
            c.append(Sq_Model.c.val)
            label = label + ", c: %.2f" % (c[-1])

        # Plot
        fig = plt.figure(figsize=[8, 6], dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X, Y, color="k")
        ax.plot(X_plot, Sq_Model(X_plot), color="k", label=label)
        ax.set_xlabel("-".join([rat_id, rat_dir[rat_id]]))
        ax.grid()
        ax.legend()
        plt.savefig(os.path.join(pic_outdir, "-".join([rat_id, rat_dir[rat_id]])+".png"))
        plt.close()

    if method == "linear":
        column_A = ["SSE", "a", "b"]
        # column_B = column_A + ["C"]
        column_names = ["RatID"] + ["SqA+SqB_"+item for item in column_A]
        result_df = pd.DataFrame(list(zip(rat_ID, SSE, a, b)),columns=column_names)
        out_filename = os.path.join(Rootpath,"LinearFitResult_" + group + ".csv")

    if method == "sigmoid":
        column_A = ["SSE", "a", "b", "c"]
        # column_B = column_A + ["C"]
        column_names = ["RatID"] + ["SqB_Proportion_"+item for item in column_A]
        result_df = pd.DataFrame(list(zip(rat_ID, SSE, a, b, c)),
                                 columns=column_names)
        #out_filename = os.path.join(Rootpath, "SigmoidFitResult_" + group + ".csv")
        out_filename = os.path.join(Rootpath, "SigmoidFitSqB_" + group + ".csv")

    with open(out_filename, "w") as f:
        f.write(result_df.to_csv(index=False))