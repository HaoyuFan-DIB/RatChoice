import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import collections
pd.options.mode.chained_assignment = None  # default='warn'

from Linear import Linear
from Sigmoid import Sigmoid
from sherpa.optmethods import NelderMead
from sherpa.fit import Fit
from sherpa.data import Data1D

def parse_column(columns):
    out_dic = collections.OrderedDict()
    for column in columns:
        sq, roi, rat_id = column.split("-")
        if rat_id not in out_dic.keys():
            out_dic[rat_id] = roi
    return out_dic


def parse_column_reverse(columns):
    out_dic = collections.OrderedDict()
    for column in columns:
        sq, roi, rat_id = column.split("-")
        if roi not in out_dic.keys():
            out_dic[roi] = [rat_id]
        else:
            out_dic[roi].append(rat_id)

    for key in out_dic.keys():
        out_dic[key] = np.unique(out_dic[key])

    return out_dic


Rootpath = '/media/zhemengwu/Gigantic Data/Rat/Weibull/Individual'
groups = ["dHPC", "Ent", "vHPC"]
outliers = [[], [15, 20], []]

for group, outlier in zip(groups, outliers):
    df = pd.read_csv(os.path.join(Rootpath, group.join(["SingleContext_", ".csv"])))

    # Remove outliers
    columns = df.columns[2::]
    rat_dic = parse_column(columns)
    for out_rat in outlier:
        columns2drop = ["SqA-%s-R%i" % (rat_dic["R%i" % out_rat], out_rat)] * 2
        columns2drop[1] = columns2drop[1].replace("SqA", "SqB")
        df = df.drop(columns=columns2drop)
        rat_dic.pop("R%i" % out_rat)

    # get result df for mean and stdev
    columns = df.columns[2::]
    roi_dic = parse_column_reverse(columns)
    result_df = df[["Session", "Date"]]
    for roi in roi_dic.keys():
        rats = roi_dic[roi]
        n_rats = len(rats)

        for sq in ["SqA", "SqB"]:
            column2use = ["-".join([sq, roi, item]) for item in rats]
            mean = df[column2use].mean(axis=1)
            result_df["-".join([sq, roi])] = mean
            stdev = df[column2use].std(axis=1) / np.sqrt(n_rats)
            result_df["-".join([sq, roi, "err"])] = stdev

    # example for Linear, you can do the rest
    colormap = ["r", "b"]
    for color, condition in zip(colormap, roi_dic.keys()):
        if "mcherry" in condition.lower() or "c21" in condition.lower():
            color = "r"
        else:
            color = "b"

        sqa = result_df["-".join(["SqA", condition])].to_numpy()
        sqa_error = result_df["-".join(["SqA", condition, "err"])].to_numpy()
        sqb = result_df["-".join(["SqB", condition])].to_numpy()
        sqb_error = result_df["-".join(["SqB", condition, "err"])].to_numpy()

        # SqA + SqB = Linear
        # SqB = Sigmoid
        # So SqA = Linear - Sigmoid

        X = df["Session"].to_numpy()
        data2fit = Data1D("data2fit", X, sqa)
        model_linear = Linear(name="linear_part")
        model_sigmoid = Sigmoid(name="s_part")
        model2fit = model_linear * (1 - model_sigmoid)
        fit = Fit(data2fit, model2fit, method=NelderMead())
        result = fit.fit()

        x_plot = np.arange(start=np.min(X), stop=np.max(X), step=0.01)
        plt.scatter(X, sqa, color=color)
        for x, y, err in zip(X, sqa, sqa_error):
            plt.plot([x, x], [y + err, y - err], color=color, linestyle=":")
        plt.plot(X, model2fit(X), color=color, label=condition)

    plt.grid(color="0.8")
    plt.legend()
    plt.xlabel(group)
    # plt.ylim([0, 0.12])
    plt.savefig(os.path.join("/media/zhemengwu/Gigantic Data/Rat/Weibull/Plot_RawData/SqA_CombinedModel",
                             group+".png"))
    plt.show()

