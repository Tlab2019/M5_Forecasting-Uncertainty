import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import gc
import os
from joblib import Parallel, delayed

import scipy.stats  as stats
from tqdm import tqdm

########################################################################
# functions for transforming point predictoin to uncertainty prediction
# refer to https://www.kaggle.com/kneroma/from-point-to-uncertainty-prediction
########################################################################
class TRANSFORM_v1(object):
    
    def __init__(self):
        self.cols = [f"F{i}" for i in range(1, 29)]
        self.qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])   

    def logit_func(self,x,a):
        return np.log(x/(1-x))*a

    def quantile_coefs(self,q):
        return ratios.loc[q].values


    
    def get_group_preds(self,pred, level):


        df = pred.groupby(level)[self.cols].sum()

        q = np.repeat(qs, len(df))


        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        df[self.cols] *= logit_func(q,0.65)[:, None] # トジット変換


        if level != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]


        df = df[["id"]+list(self.cols)]
        return df

    def get_couple_group_preds(self, pred, level1, level2):
        df = pred.groupby([level1, level2])[self.cols].sum()
        q = np.repeat(self.qs, len(df))
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        df[self.cols] *= logit_func(q,0.65)[:, None]
        df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 
                    zip(df[level1].values,df[level2].values, q)]
        df = df[["id"]+list(self.cols)]
        return df


    def point2unc(self,sub):

        levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
        couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                    ("state_id", "cat_id"),("store_id","cat_id")]
        self.cols = [f"F{i}" for i in range(1, 29)]

        df = []
        for level in levels :
            df.append(self.get_group_preds(sub, level))

        for level1,level2 in couples:
            df.append(self.get_couple_group_preds(sub, level1, level2))


        df = pd.concat(df, axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df,df] , axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                            "_validation$", "_evaluation")

        return df
    

########################################################################
# functions for transforming point predictoin to uncertainty prediction
# refer to https://www.kaggle.com/kneroma/from-point-to-uncertainty-prediction
# Updated by T.Yonezu
########################################################################
class TRANSFORM_v2(object):
    
    def __init__(self):
        
        self.cols = [f"F{i}" for i in range(1, 29)]
        self.qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]) 
        
        # まとまりごとに正規分布の分散を変える.
        self.level_coef_dict = {
                                "id": self.get_ratios(coef=0.3), 
                                "item_id": self.get_ratios(coef=0.15),
                                "dept_id": self.get_ratios(coef=0.08), 
                                "cat_id": self.get_ratios(coef=0.07),
                                "store_id": self.get_ratios(coef=0.08), 
                                "state_id": self.get_ratios(coef=0.07), 
                                "_all_": self.get_ratios(coef=0.05),
                                ("state_id", "item_id"): self.get_ratios(coef=0.19),
                                ("state_id", "dept_id"): self.get_ratios(coef=0.1),
                                ("store_id","dept_id") : self.get_ratios(coef=0.11), 
                                ("state_id", "cat_id"): self.get_ratios(coef=0.08),
                                ("store_id","cat_id"): self.get_ratios(coef=0.1)
                               }
        
          
    def get_ratios(self,coef=0.15):
        qs2 = np.log(self.qs/(1-self.qs))*coef
        ratios = stats.norm.cdf(qs2)
        ratios /= ratios[4]
        ratios = pd.Series(ratios, index=self.qs)
        return ratios.round(3)

    def quantile_coefs(self,q, level):
        ratios = self.level_coef_dict[level]
        return ratios.loc[q].values

    def get_group_preds(self,pred, level):
        df = pred.groupby(level)[self.cols].sum()
        q = np.repeat(self.qs, len(df))
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        df[self.cols] *= self.quantile_coefs(q, level)[:, None]
        
        if level != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        
        df = df[["id"]+list(self.cols)]
        
        return df

    def get_couple_group_preds(self,pred, level1, level2):
        
        df = pred.groupby([level1, level2])[self.cols].sum()
        
        q = np.repeat(self.qs, len(df))
        
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        df[self.cols] *= self.quantile_coefs(q, (level1, level2))[:, None]
        df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in zip(df[level1].values,df[level2].values, q)]
        df = df[["id"]+list(self.cols)]
        
        return df


    def point2unc(self,sub):
        levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
        couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                    ("state_id", "cat_id"),("store_id","cat_id")]

        df = []
        for level in levels:
            df.append(self.get_group_preds(sub, level))

        for level1,level2 in couples:
            df.append(self.get_couple_group_preds(sub, level1, level2))

        df = pd.concat(df, axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df,df] , axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                            "_validation$", "_evaluation")

        return df
    

########################################################################
# functions for transforming point predictoin to uncertainty prediction
# refer to https://www.kaggle.com/kneroma/from-point-to-uncertainty-prediction
# Updated by T.Yonezu
########################################################################
from scipy.stats import poisson
from multiprocessing import Pool, cpu_count

class TRANSFORM_v3(object):
    
    def __init__(self):
        
        self.cols = [f"F{i}" for i in range(1, 29)]
        self.qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]) 
        
        # まとまりごとに正規分布の分散を変える.
        self.level_coef_dict = {
                                "id": self.get_ratios(coef=0.3), 
                                "item_id": self.get_ratios(coef=0.15),
                                "dept_id": self.get_ratios(coef=0.08), 
                                "cat_id": self.get_ratios(coef=0.07),
                                "store_id": self.get_ratios(coef=0.08), 
                                "state_id": self.get_ratios(coef=0.07), 
                                "_all_": self.get_ratios(coef=0.05),
                                ("state_id", "item_id"): self.get_ratios(coef=0.19),
                                ("state_id", "dept_id"): self.get_ratios(coef=0.1),
                                ("store_id","dept_id") : self.get_ratios(coef=0.11), 
                                ("state_id", "cat_id"): self.get_ratios(coef=0.08),
                                ("store_id","cat_id"): self.get_ratios(coef=0.1)
                               }
    
    
    
    def get_ratios(self,coef=0.15):
        qs2 = np.log(self.qs/(1-self.qs))*coef
        ratios = stats.norm.cdf(qs2)
        ratios /= ratios[4]
        ratios = pd.Series(ratios, index=self.qs)
        return ratios.round(3)

    def quantile_coefs(self,q, level):
        ratios = self.level_coef_dict[level]
        return ratios.loc[q].values
    
    def get_poisson_ppf(self,input):
        return poisson.ppf(input[0],input[1])
    
    def get_group_preds(self,pred, level):
        df = pred.groupby(level)[self.cols].sum()
        q = np.repeat(self.qs, len(df))
        df = pd.concat([df]*len(self.qs), axis=0, sort=False)
        df.reset_index(inplace = True)
        
        for c in tqdm(self.cols):
            p = Pool(cpu_count()) # プロセス数を4に設定
            df[c] = p.map(self.get_poisson_ppf, np.array([q,df[c].values]).T)
            p.close()
        
        if level != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        
        df = df[["id"]+list(self.cols)]
        
        return df

    def get_couple_group_preds(self,pred, level1, level2):
        
        df = pred.groupby([level1, level2])[self.cols].sum()
        
        q = np.repeat(self.qs, len(df))
        
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        
        for c in tqdm(self.cols):
            p = Pool(cpu_count()) # プロセス数を4に設定
            df[c] = p.map(self.get_poisson_ppf, np.array([q,df[c].values]).T)
            p.close()
            
        df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in zip(df[level1].values,df[level2].values, q)]
        df = df[["id"]+list(self.cols)]
        
        return df


    def point2unc(self,sub):
        levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
        couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                    ("state_id", "cat_id"),("store_id","cat_id")]

        df = []
        for level in levels:
            df.append(self.get_group_preds(sub, level))

        for level1,level2 in couples:
            df.append(self.get_couple_group_preds(sub, level1, level2))

        df = pd.concat(df, axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df,df] , axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                            "_validation$", "_evaluation")

        return df