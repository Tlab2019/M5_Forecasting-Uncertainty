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

def encode_onehot(df, cols):
    df = pd.get_dummies(df, columns=cols, sparse=True)
    
    return df

quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']
days = range(1, 1914)
val_eval = ['validation', 'evaluation']

def CreateSales( train_sales,name_list, group):
    '''This function returns a dataframe (sales) on the aggregation level given by name list and group'''
    
    rows_ve = [(name + "_X_" + str(q) + "_" + ve, str(q)) for name in name_list for q in quants for ve in val_eval]
    time_series_columns = [c for c in train_sales.columns if "d_" in c]
    time_series_columns += [c for c in train_sales.columns if "F" in c]
    sales = train_sales.groupby(group)[time_series_columns].sum() #would not be necessary for lowest level
    return sales


def createTrainSet(sales_train_s,train_sales, name, group_level, X = False):
    sales_total = CreateSales(train_sales,name, group_level)
    if(X == True):
        sales_total = sales_total.rename(index = lambda s:  s + '_X')
    sales_train_s = sales_train_s.append(sales_total)
    return(sales_train_s)

def get_agg_df(train_sales):
    total = ['Total']
    train_sales['Total'] = 'Total'
    train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id
    train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id
    train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id
    train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id
    train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id
    train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id
    total = ['Total']
    states = ['CA', 'TX', 'WI']
    num_stores = [('CA',4), ('TX',3), ('WI',3)]
    stores = [x[0] + "_" + str(y + 1) for x in num_stores for y in range(x[1])]
    cats = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
    num_depts = [('FOODS',3), ('HOBBIES',2), ('HOUSEHOLD',2)]
    depts = [x[0] + "_" + str(y + 1) for x in num_depts for y in range(x[1])]
    state_cats = [state + "_" + cat for state in states for cat in cats]
    state_depts = [state + "_" + dept for state in states for dept in depts]
    store_cats = [store + "_" + cat for store in stores for cat in cats]
    store_depts = [store + "_" + dept for store in stores for dept in depts]
    prods = list(train_sales.item_id.unique())
    prod_state = [prod + "_" + state for prod in prods for state in states]
    prod_store = [prod + "_" + store for prod in prods for store in stores]
    cols = [i for i in train_sales.columns if i.startswith('F')]
    sales_train_s = train_sales[cols]
    sales_train_s = pd.DataFrame()
    sales_train_s = createTrainSet(sales_train_s,train_sales, total, 'Total', X=True) #1
    sales_train_s = createTrainSet(sales_train_s, train_sales,states, 'state_id', X=True) #2
    sales_train_s = createTrainSet(sales_train_s,train_sales, stores, 'store_id', X=True) #3
    sales_train_s = createTrainSet(sales_train_s,train_sales, cats, 'cat_id', X=True) #4
    sales_train_s = createTrainSet(sales_train_s,train_sales, depts, 'dept_id', X=True) #5
    sales_train_s = createTrainSet(sales_train_s,train_sales, state_cats, 'state_cat') #6
    sales_train_s = createTrainSet(sales_train_s,train_sales, state_depts, 'state_dept') #7
    sales_train_s = createTrainSet(sales_train_s,train_sales, store_cats, 'store_cat') #8
    sales_train_s = createTrainSet(sales_train_s,train_sales, store_depts, 'store_dept') #9
    sales_train_s = createTrainSet(sales_train_s,train_sales, prods, 'item_id', X=True) #10
    sales_train_s = createTrainSet(sales_train_s,train_sales, prod_state, 'state_item') #11
    sales_train_s = createTrainSet(sales_train_s,train_sales, prod_store, 'item_store')
    sales_train_s['id'] = sales_train_s.index
    return(sales_train_s)