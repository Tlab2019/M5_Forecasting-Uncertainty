import numpy as np
import glob
import os
import time

path = os.path.join("submission_uncertainty","Prophet_v1_gaussian_eval","*.pickle")

while True:
    num = len(glob.glob(path))
    print("filenum:",str(num),"|","completed:",round(num/42840*100,2),"%")
    time.sleep(5)