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
class Weibull_PDF(ArithmeticModel):
    # PDF of Weibull distribution
    # Y = (k/lambda_0) * (x/lambda_0)^(k-1) * exp[-(x/lambda_0)^k] (x >= 0)
    # Y = 0   (x < 0)

    # We do not require normalization, so we need to add a scaling factor A to the distribution
    # Parameters are thus as follows:
    # X: Independent variable, time in the form of session no. or date in our case
    # Y: Dependent variable, accuracy or inaccuracy in our case
    # A: Scaling factor for y, positive
    # k: shape parameter, positive
    # lambda_0: Scaling factor for x, like time-scale, positive

    def __init__(self, name=None, A=1.0, k=1.0, lambda_0=1.0, delay=0.0):
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
        #assert np.min(x) >= 0, "Input X must be positive"

        # Y = (k/lambda_0) * (x/lambda_0)^(k-1) * exp[-(x/lambda_0)^k] (x >= 0)
        A, k, lambda_0, delay = pars
        x = np.abs((x - delay) / lambda_0)
        y = 1 - A * (k / lambda_0) * x ** (k-1) * np.exp(-1 * x**k)
        return y


if __name__ == "__main__":

    # loading data
    data_filename = "/media/zhemengwu/Gigantic Data/Rat/Weibull/SingleContextData-dHPC.csv"
    df = pd.read_csv(data_filename, sep=",")
    conditions = df.columns[2::2]

    # For now, test 'SqA-vHPC-SAL'
    condition = "SqA-mCherry"
    X = df["Session"].to_numpy() - 1
    Y = df[condition].to_numpy()
    Err = df[condition + "-err"].to_numpy()

    data2fit = Data1D('data2fit', X, Y)
    model2fit = Weibull_PDF(name="Weibull", k=1.0, lambda_0=20)
    # model2fit.delay.freeze()
    fit = Fit(data2fit, model2fit, method=NelderMead())
    result = fit.fit()
    # error = fit.est_errors()

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
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title(condition)
    # plt.savefig(os.path.join("/media/zhemengwu/Gigantic Data/Weibull",condition+".png"))
    plt.show()
