import numpy as np
import pandas as pd

def check_columns(necessary_cols,cols):
    
    cols = set(cols) # make set
    
    lack_cols = [c for c in necessary_cols if c not in cols]
    
    print("-- column check completed --")
    if len(lack_cols) == 0:
        print("  columns are satisfied")
        return True
    else:
        print("  !!columns are lacked!!")
        print("   lacked columns:",lack_cols)
        return False