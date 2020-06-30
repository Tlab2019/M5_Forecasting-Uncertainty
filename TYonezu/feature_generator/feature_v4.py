import pandas as pd
import glob
import os
import numpy as np
import gc
import copy

from .function import *
from sklearn.preprocessing import LabelEncoder
from .myUtils import *

################################
# make features using by H.Kato
################################
class FeaturesMaker_v4(object):
    
    def __init__(self,target_col):
        self.name = "features_ver4"
        self.feature_exp = "features from Stat-of-the-art NoteBook and one-hot encoded store_id"
        
        self.target_col = target_col
        self.necessary_col =  ["id"] + ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + ["data_part"] + [target_col]
        
    def make_feature(self,df):
        
        
        # check existstance of necessary columns
        if check_columns(self.necessary_col,df.columns):
            
            # make lag features
            #for lag in [1,2,3,6,12,24,36]:
            for lag in [28,35,42,49,56]:
                df['sold_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)[self.target_col].shift(lag).astype(np.float16)

            # save target values temporaly
            target_values_tmp = copy.copy(df[self.target_col])
            mask = (df["data_part"] == "train")|(df["data_part"] == "validation")
            df.loc[~mask,self.target_col] = np.nan


            # make State-of-the-art feature
            df['item_sold_avg'] = df.groupby('item_id')[self.target_col].transform('mean').astype(np.float16)
            df['state_sold_avg'] = df.groupby('state_id')[self.target_col].transform('mean').astype(np.float16)
            df['store_sold_avg'] = df.groupby('store_id')[self.target_col].transform('mean').astype(np.float16)
            df['cat_sold_avg'] = df.groupby('cat_id')[self.target_col].transform('mean').astype(np.float16)
            df['dept_sold_avg'] = df.groupby('dept_id')[self.target_col].transform('mean').astype(np.float16)
            df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])[self.target_col].transform('mean').astype(np.float16)
            df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])[self.target_col].transform('mean').astype(np.float16)
            df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])[self.target_col].transform('mean').astype(np.float16)
            df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])[self.target_col].transform('mean').astype(np.float16)
            df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])[self.target_col].transform('mean').astype(np.float16)
            df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])[self.target_col].transform('mean').astype(np.float16)
            df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])[self.target_col].transform('mean').astype(np.float16)

            #df['rolling_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])[self.target_col].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)
            df['expanding_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])[self.target_col].transform(lambda x: x.expanding(2).mean()).astype(np.float16)

            df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])[self.target_col].transform('mean').astype(np.float16)
            df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])[self.target_col].transform('mean').astype(np.float16)
            #df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)
            df.drop(['daily_avg_sold','avg_sold'],axis=1,inplace=True)
            
            # label encoding
            cols = ['item_id', 'dept_id', 'cat_id', 'state_id']
            df = label_encode(df, cols=cols)
            df = onehot_encode(df, cols=["store_id"])
            
            
            # split train and test
            df = df.set_index(["id","d"],drop=True)
            
            features = [c for c in df.columns if c not in set(["data_part",self.target_col,"date","weekday"])]
            
            print("-- ",self.name," --")
            print("dim:",len(features))
            print("N:",len(df))
            print("-----------------")
            
            df[self.target_col] = target_values_tmp.values
            
            del target_values_tmp
            gc.collect()
            
            return {sub[0]:(sub[1][features],sub[1][self.target_col]) for sub in df.groupby(by="data_part")}
        
        else:
            return False