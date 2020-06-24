import pandas as pd
import glob
import os
import numpy as np
import gc

from .function import *
from sklearn.preprocessing import LabelEncoder

def label_encode(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    #
    return df


class FeaturesMaker_v1(object):
    
    def __init__(self,target_col):
        self.name = "features_ver1"
        self.feature_exp = "simple features used label encoded [item,store,dept,cat,event] and prices"
        
        self.target_col = target_col
        self.necessary_col =  ["id"] + ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + ["data_part"] + [target_col]
        
    def make_feature(self,df):
        
        # check existstance of necessary columns
        if check_columns(self.necessary_col,df.columns):
            
            # calendar and event information
            calendar = pd.read_csv(os.path.join("rawdata","calendar.csv"))
            
            # price information
            sell_prices = pd.read_csv(os.path.join("rawdata","sell_prices.csv"))
            sell_prices = sell_prices.groupby(by=["item_id","store_id"]).agg({"sell_price":["median","mean","max","min"]})
            sell_prices = sell_prices.reset_index()
            sell_prices.columns = ["item_id","store_id","price-median","price-mean","price-max","price-min"]
            
            # concat information
            df = pd.merge(df,calendar,on=["d"],how="left") # カレンダー情報
            df = pd.merge(df,sell_prices,on=["item_id","store_id"],how="left") # 価格情報
            
            del sell_prices, calendar
            gc.collect()
            
            # year,month,wday
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["wday"] = df["date"].dt.weekday
            
            # label encoding
            cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
            df = label_encode(df, cols=cols)
            
            
            # split train and test
            df = df.set_index(["id","d"],drop=True)
            
            features = [c for c in df.columns if c not in set(["data_part",self.target_col,"date","weekday"])]
            
            print("-- ",self.name," --")
            print("dim:",len(features))
            print("N:",len(df))
            print("-----------------")
            
            return {sub[0]:(sub[1][features],sub[1][self.target_col]) for sub in df.groupby(by="data_part")}
        
        else:
            return False