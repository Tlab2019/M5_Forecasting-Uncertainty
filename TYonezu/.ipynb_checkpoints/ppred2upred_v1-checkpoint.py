import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import gc
import os
from joblib import Parallel, delayed

########################################################################
# functions for transforming point predictoin to uncertainty prediction
# refer to https://www.kaggle.com/kneroma/from-point-to-uncertainty-prediction
########################################################################
def logit_func(x,a):
    return np.log(x/(1-x))*a

def quantile_coefs(q):
    return ratios.loc[q].values


# levelでgroupbyしてsumを取る. 
cols = [f"F{i}" for i in range(1, 29)]
qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])    
def get_group_preds(pred, level):
    

    df = pred.groupby(level)[cols].sum()
    
    q = np.repeat(qs, len(df))
    
    
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= logit_func(q,0.65)[:, None] # トジット変換
    
    
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        
        
    df = df[["id"]+list(cols)]
    return df

def get_couple_group_preds(pred, level1, level2):
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= logit_func(q,0.65)[:, None]
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    return df


def point2unc(sub):
    
    levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
    couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                ("state_id", "cat_id"),("store_id","cat_id")]
    cols = [f"F{i}" for i in range(1, 29)]

    df = []
    for level in levels :
        df.append(get_group_preds(sub, level))
        
    for level1,level2 in couples:
        df.append(get_couple_group_preds(sub, level1, level2))
    
    
    df = pd.concat(df, axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df,df] , axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                        "_validation$", "_evaluation")
    
    return df