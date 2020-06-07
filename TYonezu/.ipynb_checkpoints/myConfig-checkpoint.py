import os
from myTransform import *

data_config = {
    "data_v1":"standard label encoding and price"
}


feature_config = {
    "features_v1":[
                   'item_id',
                   'dept_id',
                   'cat_id',
                   'store_id',
                   'state_id',
                   #'d',
                   'wday',
                   'month',
                   'year',
                   'event_name_1',
                   'event_type_1',
                   'event_name_2',
                   'event_type_2',
                   'snap_CA',
                   'snap_TX',
                   'snap_WI',
                   'price-median',
                   'price-mean',
                   'price-max',
                   'price-min'
                   ],
    
    "features_v2":None
}

