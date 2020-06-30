#!/usr/bin/env python
# coding: utf-8

# ### **Regression by Prophet**
# make uncertainty prediction by Prophet regression

# In[1]:
import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
from tqdm import tqdm

from myConfig import *
import gc
from myUtils import *
from myProphet import *

from scipy.stats import norm
from multiprocessing import Pool, cpu_count

import datetime as dt

dir = os.path.join("submission_uncertainty","Prophet_v1_gaussian_eval")
if not(os.path.exists(dir)):
    os.makedirs(dir)

def Prophet_Gauss_Upred(input):
    
    #print(input)
    col = input["col"]
    ts = input["ts"]
    start = input["start"]
    end = input["end"]
    
    dir = os.path.join("submission_uncertainty","Prophet_v1_gaussian_eval")
    file_name = os.path.join(dir,col+".pickle")
    qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]) 
    F_cols = [f"F{i}" for i in range(1, 29)]
    
    if os.path.exists(file_name):
        pass
    else:
        para = pd.DataFrame({"quantile":np.repeat(qs,len(F_cols)),
                             "d":np.tile(F_cols,len(qs))})



        # prophet regression
        model = myProphet()
        model.fit(y=ts)
        pred = model.predict(start=start,end=end,freq="1D",mode="all")

        # 正規分布の平均推定
        para["mean"] = np.tile(pred["yhat"].values, len(qs))

        # 正規分布の分散推定
        sigU = (pred["yhat_upper"] - pred["yhat"])/norm.ppf(0.8,loc=0,scale=1)
        sigL = -(pred["yhat_lower"] - pred["yhat"])/norm.ppf(0.8,loc=0,scale=1)
        var = (sigU+sigL)/2
        para["std"] = np.tile(var.values, len(qs))

        # Quantile regression
        upred = []
        for q,mean,std in para[["quantile","mean","std"]].values:
            upred.append(norm.ppf(q,loc=mean,scale=std))
        para["q_pred"] = upred

        # make pivot table
        para = pd.pivot(para,
                        index="quantile",
                        columns="d",
                        values="q_pred")
        para = para.reset_index()

        para["data_part"] = "validation"
        para["id"] = col
        para["quantile"] = [f"{q:.3f}" for q in para["quantile"].values]

        para["id"] = para["id"].str.cat(para["quantile"],sep="_")
        para["id"] = para["id"].str.cat(para["data_part"],sep="_")
        para = para.drop(columns=["quantile","data_part"])

        para[["id"]+F_cols].to_pickle(file_name)
    
    print(col," finished")
    
    return True

if __name__ == "__main__":

    data_path = os.path.join("mydata","agg_train_eval.pickle")
    data = pd.read_pickle(data_path)

    cols = [c for c in data.columns if "d_" in c]
    data = data[cols]
    data = data.T
    data.index.name = "d"
    data = data.reset_index()

    calendar = pd.read_csv(os.path.join("rawdata","calendar.csv"),parse_dates=[0])
    data = pd.merge(data,calendar[["d","date"]],on="d",how="left")
    data = data.set_index("date",drop=True)
    data = data.drop(columns={"d"})

    # ### **Prophet Regression**
    # https://facebook.github.io/prophet/docs/quick_start.html

    # Forecast days
    
    start = dt.datetime(2016,4,25) + dt.timedelta(days=28)
    end = start + dt.timedelta(days=27)

    print("Forecast ",start,"<->",end)

    input = []
    for col in tqdm(data.columns):  
        tmp = {"col":col,"ts":data[col],"start":start,"end":end}
        input.append(tmp)

    p = Pool(cpu_count()) # プロセス数を8に設定
    p.map(Prophet_Gauss_Upred, input)
    p.close()