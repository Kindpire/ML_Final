import pyprind
import pandas as pd
import os
import numpy as np
import csv
import operator

# export train csv file
# better to find the num of file by itself, but i'm lazy
pbar = pyprind.ProgBar(25000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()

for l in ('pos','neg'):
    path = './data/train/%s' % l
    for file in os.listdir(path):
        with open(os.path.join(path,file),'r') as infile:
            txt = infile.read()
        df = df.append([[txt, labels[l]]],ignore_index=True)
        pbar.update()
df.columns = ['review','sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data_train.csv',index=False)

# export test csv file

pbar = pyprind.ProgBar(11000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()

path = './data/test'
counter = 0
for file in os.listdir(path):
    if not file.startswith('.'):
        counter+=1
        with open(os.path.join(path,file),'r') as infile:
            txt = infile.read()
        df = df.append([[file[:-4],txt]],ignore_index=True)
        pbar.update()
    else:
        print("i got a bad one")

df.columns = ['file','review']
df.to_csv('./data/test_temp.csv',index=False)

with open('./data/test_temp.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    reader.next()
    sortedlist = sorted(reader, key=lambda t: int( t[0] ), reverse=False)

with open('./data/movie_data_test.csv','w') as sortedfile:
    fieldnames = ['file', 'review']
    writer = csv.writer(sortedfile)
    writer.writerow(fieldnames)
    writer.writerows(sortedlist)
