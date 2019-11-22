import os, sys
from time import time, localtime, strftime

path = './exp/log/Saved/Pexod/15_11_03/'

txt = "Horizon2 = [\n"

files = os.listdir(path)
files.sort()
for name in files:
    if "" in name and "" in name:
        txt += '['
        replicates = os.listdir(path+name)
        for replicate in replicates:
            files = os.listdir(path+name+"/"+replicate)
            for file in files:
                if "2019-" in file:
                    txt += "\'"+path+name+"/"+replicate+"/"+file+"/logs.mat"+"\', "
        txt += '],\n'
txt += ']\n'

with open("./exp/log/Saved/"+strftime("%Y-%m-%d--%H:%M:%S", localtime()), 'w') as f:
    f.write(txt)
