import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from Weibull_cdf import Weibull_cdf
from Weibull_pdf import Weibull_PDF
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
    A_k, A_lambda_0, A_A, A_delay, A_SSE = [], [], [], [], []
    # parameters for SqB
    B_k, B_lambda_0, B_A, B_delay, B_C, B_SSE = [], [], [], [], [], []

    for rat_id in rat_dir.keys():
        # Gather data
        SqA_column = "-".join(["SqA", rat_dir[rat_id], rat_id])
        SqB_column = SqA_column.replace("SqA", "SqB")
        Y_SqA = df[SqA_column].to_numpy()
        Y_SqB = df[SqB_column].to_numpy()

        # SqA Model: Weibull_PDF
        SqA_Model = Weibull_PDF(name="Weibull", k=1.0, lambda_0=20)
        SqA_Model.delay.freeze()
        SqA_Data = Data1D("SqA_Data", X, Y_SqA)
        fit = Fit(SqA_Data, SqA_Model, method=NelderMead())
        result_SqA = fit.fit()

        # SqB Model: Weibull_CDF
        SqB_Model = Weibull_cdf(base=np.e)
        SqB_Model.C.val = 0.0
        SqB_Model.C.freeze()
        SqB_Model.delay.val = 0.0
        SqB_Model.delay.freeze()
        SqB_Data = Data1D("SqB_Data", X, Y_SqB)
        fit = Fit(SqB_Data, SqB_Model, method=NelderMead())
        result_SqB = fit.fit()

        # Output fit result
        rat_ID.append(rat_id)

        A_A.append(SqA_Model.A.val)
        A_k.append(SqA_Model.k.val)
        A_lambda_0.append(SqA_Model.lambda_0.val)
        A_delay.append(SqA_Model.delay.val)
        A_SSE.append(np.sum((Y_SqA - SqA_Model(X))**2))

        B_A.append(SqB_Model.A.val)
        B_k.append(SqB_Model.k.val)
        B_lambda_0.append(SqB_Model.lambda_0.val)
        B_delay.append(SqB_Model.delay.val)
        B_C.append(SqB_Model.C.val)
        B_SSE.append(np.sum((Y_SqB - SqB_Model(X))**2))


        # Plot
        fig = plt.figure(figsize=[8, 6], dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X, Y_SqA, color="r")
        ax.plot(X_plot, SqA_Model(X_plot), color="r", label="SqA, SSE=%.2f" % A_SSE[-1])
        ax.scatter(X, Y_SqB, color="b")
        ax.plot(X_plot, SqB_Model(X_plot), color="b", label="SqB, SSE=%.2f" % B_SSE[-1])
        ax.set_xlabel("-".join([rat_id, rat_dir[rat_id]]))
        ax.grid()
        ax.legend()
        plt.savefig(os.path.join(pic_outdir, "-".join([rat_id, rat_dir[rat_id]])+".png"))
        plt.close()

    column_A = ["SSE", "A", "k", "lambda_0", "delay"]
    column_B = column_A + ["C"]
    column_names = ["RatID"] + ["SqA_"+item for item in column_A] + ["SqB_"+item for item in column_B]
    result_df = pd.DataFrame(list(zip(rat_ID,
                                      A_SSE, A_A, A_k, A_lambda_0, A_delay,
                                      B_SSE, B_A, B_k, B_lambda_0, B_delay, B_C)),
                             columns=column_names)
    with open(os.path.join(Rootpath, "FitResult_" + group + ".csv"), "w") as f:
        f.write(result_df.to_csv(index=False))