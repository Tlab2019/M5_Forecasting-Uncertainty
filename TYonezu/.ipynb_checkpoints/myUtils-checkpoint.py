import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import gc
import os
from joblib import Parallel, delayed

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    #
    return df


########################################################################
# functions for transforming point predictoin to uncertainty prediction
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