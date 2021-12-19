import numpy as np
import pandas as pd
import os

class SingleContextDataProcessor():

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
                if "035" in data_dict.keys():
                    self.Corr.append(0)
                else:
                    self.Corr.append(1)
            else:
                self.Resp.append("R")
                self.RT.append((float(data_dict["003"]) - float(data_dict["026"]))/100)
                if "037" in data_dict.keys():
                    self.Corr.append(0)
                else:
                    self.Corr.append(1)

if __name__ == '__main__':
    df_all = []
    for i in range(8):
        df_all.append(pd.DataFrame(columns=["Date", "Trial", "Context", "Sequence", "Response", "Correct", "RT"]))
    subject_name_dict = {17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 26:24}

    rootpath = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Data_6th"
    for root, dirs, files in os.walk(rootpath):
        files.sort()
        for file in files:
            print(file)
            if file[0] == "!":
                processor = SingleContextDataProcessor(os.path.join(rootpath, file))
                df = processor.getData()
                if df is not None:
                    rat_ID = subject_name_dict[int(file[-2:])]
                    rat_idx = rat_ID - 17 # 0:7, idx in df_all
                    df_all[rat_idx] = df_all[rat_idx].append(df)
            else:
                print("{f} skipped!".format(f=file))

    savepath = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Result"
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for i, item in enumerate(df_all):
        filename = "SingleContext_Rat{i}.csv".format(i=i+17)
        with open(os.path.join(savepath, filename), "w") as f:
            f.write(item.to_csv(index=False))