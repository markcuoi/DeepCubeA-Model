import pickle
import os
import numpy as np

data = [26,43,33,28,4,1,45,5,2,29,32,35,16,13,25,17,14,42,6,39,36,7,22,19,8,50,47,9,10,11,48,31,41,51,34,44,27,37,20,30,40,23,53,12,15,38,3,0,46,49,52,18,21,24]

file_name = "state_0.pkl"
f = open(file_name,"wb")
pickle.dump(data,f)