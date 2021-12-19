import pandas as pd
import numpy as np
import datetime
import os

class DoubleContextDataProcessor():

    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename)

    def getData(self):
        date = self._checkFile()
        if not date:
            return None
        else:
            self.date = date
            self.Trial = []
            self.Corr = []
            self.RT = []
            self.Sq = []
            self.Resp = []

        while True:
            line = self.f.readline().replace("\n","")
            if not line:
                break

            if line[-4:] in [".035", ".037"]:
                self._processData(line)

        self.Trial = np.arange(len(self.RT)) + 1
        self.date = [self.date] * len(self.RT)
        if "_7th" in self.filename:
            self.Context = ["Dot"] * len(self.RT)
        if "_6th" in self.filename:
            self.Context = ["Stripe"] * len(self.RT)

        df = pd.DataFrame(list(zip(self.date, self.Trial, self.Context, self.Sq, self.Resp, self.Corr, self.RT)),
                               columns=["Date", "Trial", "Context", "Sequence", "Response", "Correct", "RT"])
        return df

    def _checkFile(self):
        mm = int(self.f.readline().replace("\n",""))
        dd = int(self.f.readline().replace("\n",""))
        yy = int(self.f.readline().replace("\n",""))
        for i in range(4):
            self.f.readline()
        usr = self.f.readline().replace("\n", "")
        if usr == "Denada":
            return "-".join(["20"+"{:02d}".format(yy), "{:02d}".format(mm), "{:02d}".format(dd)])
        else:
            return False

    def _processData(self, line):
        value, trigger = line.split(sep=".")
        data_dict = {trigger: value}
        while True:
            line = self.f.readline().replace("\n", "")
            if "." not in line:
                print("invalid trial block, break out")
                print(self.filename)
                return
            value, trigger = line.split(sep=".")
            data_dict[trigger] = value
            if trigger in ["036", "038"]:
                break

        assert "035" in data_dict.keys() or "037" in data_dict.keys(), "Something Wrong in 035/037 Trigger..."
        if "035" in data_dict.keys(): self.Sq.append("A")
        else: self.Sq.append("B")

        if "001" not in data_dict.keys() and "003" not in data_dict.keys():
            self.Corr.append(-99)
            self.RT.append(-99)
            self.Resp.append("N")
        else:
            if "001" in data_dict.keys():
                self.Resp.append("L")
                self.RT.append((float(data_dict["001"]) - float(data_dict["026"]))/100)
                if "_6th" in self.filename: # stripe
                    if "035" in data_dict.keys(): #sequence A
                        self.Corr.append(0)
                    else:
                        self.Corr.append(1)
                elif "_7th" in self.filename: # dot
                    if "035" in data_dict.keys(): # sequence A
                        self.Corr.append(1)
                    else:
                        self.Corr.append(0)
            else:
                self.Resp.append("R")
                self.RT.append((float(data_dict["003"]) - float(data_dict["026"]))/100)
                if "_6th" in self.filename: # stripe
                    if "037" in data_dict.keys(): # sequence B
                        self.Corr.append(0)
                    else:
                        self.Corr.append(1)
                elif "_7th" in self.filename: #dot
                    if "037" in data_dict.keys(): # sequence B
                        self.Corr.append(1)
                    else:
                        self.Corr.append(0)


if __name__ == '__main__':
    rootpath = "/media/zhemengwu/Gigantic Data/DoubleContextSequence/Data2Use"

    df_6th = pd.DataFrame(columns=["filename", "date", "subject"])
    rootpath6 = os.path.join(rootpath, "Rat9-16_6th")
    for root, dirs, files in os.walk(rootpath6):
        files.sort()
        for file in files:
            if file[0] == "!":
                date = file[1:].split("_")[0]
                subject = int(file[-2:])
                df2add = pd.DataFrame([[file, date, subject]],
                                       columns=["filename", "date", "subject"])
                df_6th = df_6th.append(df2add, ignore_index=True)

    df_7th = pd.DataFrame(columns=["filename", "date", "subject"])
    rootpath7 = os.path.join(rootpath, "Rat9-16_7th")
    for root, dirs, files in os.walk(rootpath7):
        files.sort()
        for file in files:
            if file[0] == "!":
                date = file[1:].split("_")[0]
                subject = int(file[-2:])
                df2add = pd.DataFrame([[file, date, subject]],
                                       columns=["filename", "date", "subject"])
                df_7th = df_7th.append(df2add, ignore_index=True)

    session_all = df_6th["date"].unique()
    sub_dict = {9:17, 10:18, 11:19, 12:20,
                13:21, 14:22, 15:23, 16:26,
                17:17, 19:19, 20:20, 21:21,
                22:22, 23:23, 24:26}

    for sub in sub_dict.keys():
        print("Now working on Rat {s}".format(s=sub))
        df_out = pd.DataFrame(columns=["Date", "Trial", "Context", "Sequence", "Response", "Correct", "RT"])

        if sub == 17:
            df_6th = pd.DataFrame(columns=["filename", "date", "subject"])
            rootpath6 = os.path.join(rootpath, "Rat17-24_6th")
            for root, dirs, files in os.walk(rootpath6):
                files.sort()
                for file in files:
                    if file[0] == "!":
                        date = file[1:].split("_")[0]
                        subject = int(file[-2:])
                        df2add = pd.DataFrame([[file, date, subject]],
                                              columns=["filename", "date", "subject"])
                        df_6th = df_6th.append(df2add, ignore_index=True)

            df_7th = pd.DataFrame(columns=["filename", "date", "subject"])
            rootpath7 = os.path.join(rootpath, "Rat17-24_7th")
            for root, dirs, files in os.walk(rootpath7):
                files.sort()
                for file in files:
                    if file[0] == "!":
                        date = file[1:].split("_")[0]
                        subject = int(file[-2:])
                        df2add = pd.DataFrame([[file, date, subject]],
                                              columns=["filename", "date", "subject"])
                        df_7th = df_7th.append(df2add, ignore_index=True)


        if sub <= 16:
            df_all = [df_7th, df_6th]
            rt_all = [rootpath7, rootpath6]
        else:
            df_all = [df_6th, df_7th]
            rt_all = [rootpath6, rootpath7]

        for session in session_all:
            filename0 = df_all[0]["filename"].loc[(df_all[0]["date"] == session)
                                                  & (df_all[0]["subject"] == sub_dict[sub])]
            assert len(filename0) <= 1, \
                    "Multiple files selected for {s}, Rat {sub}, first".format(s=session, sub=sub)
            if len(filename0) == 1:
                processor = SingleContextDataProcessor(os.path.join(rt_all[0], filename0.values[0]))
                df2add = processor.getData()
                if df2add is not None:
                    df_out = df_out.append(df2add, ignore_index=True)

            filename1 = df_all[1]["filename"].loc[(df_all[1]["date"] == session)
                                                  & (df_all[1]["subject"] == sub_dict[sub])]
            assert len(filename1) <= 1, \
                    "Multiple files selected for {s}, Rat {sub}, second".format(s=session, sub=sub)
            if len(filename1) == 1:
                processor = DoubleContextDataProcessor(os.path.join(rt_all[1], filename1.values[0]))
                df2add = processor.getData()
                if df2add is not None:
                    df_out = df_out.append(df2add, ignore_index=True)

            df_all = [df_all[1], df_all[0]]
            rt_all = [rt_all[1], rt_all[0]]

        savepath = "/media/zhemengwu/Gigantic Data/DoubleContextSequence/ExpLogs"
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        with open(os.path.join(savepath, "DoubleContext_Rat{s}.csv".format(s=sub)), "w") as f:
            f.write(df_out.to_csv(index=False))

