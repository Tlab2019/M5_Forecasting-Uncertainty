import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import gc
from myUtils import *
from feature_generator import feature_v1, feature_v2
import xgboost as xgb

###################################################################
# make features
##################################################################
feature_maker = feature_v1.FeaturesMaker_v1(target_col="item_cnt")

data = pd.read_pickle(os.path.join("mydata","sales_train_eval_365.pickle"))
data = feature_maker.make_feature(data)


model_path = os.path.join("models","XGBoost_"+feature_maker.name+".mdl")

if os.path.exists(model_path):
    print("loading trained model...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
else:
    print("start training XGBoost")
    model = xgb.XGBRegressor(objective="reg:squarederror",
                             tree_method="gpu_hist",
                             random_state=0,
                             verbose=2,
                             n_estimators=1000,
                            )

    model.fit(X=data["train"][0], y=data["train"][1], 
              #sample_weight=None, 
              #base_margin=None, 
              eval_set=[data["train"],data["validation"]], 
              #eval_metric=None, 
              early_stopping_rounds=100, 
              verbose=True, 
              #xgb_model=None, 
              #sample_weight_eval_set=None
             )
    model.save_model(model_path)
print("  -- completed\n")
    
# prediction
print("start prediction")
pred_mask = data["evaluation"][1].isna()
data["evaluation"][1].loc[pred_mask] = model.predict(data["evaluation"][0])
print("  -- completed\n")


# submission 
print("start submission")
sub_path = os.path.join("submission_point","XGBoost_"+feature_maker.name+"_submission.csv")


sub_cols = ["id"] + [f"F{i}" for i in range(1, 29)]

valid = data["validation"][1]
evalu = data["evaluation"][1]

del data
gc.collect()

valid = pd.DataFrame(valid.values,
                     index=valid.index,
                     columns=[feature_maker.target_col])
evalu = pd.DataFrame(evalu.values,
                     index=evalu.index,
                     columns=[feature_maker.target_col])

valid = valid.reset_index()
evalu = evalu.reset_index()

valid = pd.pivot(valid,
                 index="id", 
                 columns="d", 
                 values=feature_maker.target_col)
evalu = pd.pivot(evalu,
                 index="id", 
                 columns="d", 
                 values=feature_maker.target_col)

valid = valid.reset_index()
evalu = evalu.reset_index()

valid.columns = sub_cols
evalu.columns = sub_cols

valid["id"] = valid["id"].str.replace("_evaluation","_validation")

pd.concat([valid,evalu]).to_csv(sub_path,index=False)
print("  -- completed")