-import os
import numpy as np
import pandas as pd

rootpath = "/media/zhemengwu/Gigantic Data/Rat/SingleContext_Hipp/Data_Control/vHPC"
for root, dirs, files in os.walk(rootpath):
    files.sort()
    for file in files:
        print(file)
        df = pd.read_csv(os.path.join(rootpath, file))
        df = df.loc[df["Trial"] <= 10]
        all_dates = df["Date"].unique()
        for date in all_dates:
            n_trail_date = len(df.loc[df["Date"] == date])
            n_correct_trail = df["Correct"].loc[df["Date"] == date].sum()
            print(date + ": %.1f" % (n_correct_trail / n_trail_date))

