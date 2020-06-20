import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .function import *


def label_encode(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df

def onehot_encode(df, cols):
    df = pd.get_dummies(df, columns=cols, sparse=True)
    
    return df


class FeaturesMaker_v2(object):
    
    def __init__(self):
        self.name = "features_ver2"
        self.feature_exp = "simple features with onehot encoded [store] and label encoded [item,dept,cat,event] and prices"
        
        self.necessary_col = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
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
            
            # year,month,wday
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["wday"] = df["date"].dt.weekday
            
            # label encoding
            df = label_encode(df, cols=['item_id', 'dept_id', 'cat_id', 'state_id','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])
            df = onehot_encode(df, col=['store_id'])
            
            print("-- ",self.name," --")
            print("N:",len(df)," d:",len(df.columns))
            print("-----------------")
            
            return df
        
        else:
            return False